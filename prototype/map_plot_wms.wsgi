#! python
"""
A WMS server for generating contour images from opendap data sources.
Not guaranteed to implement all parts of the WMS specification.

----------------------------------------------------------------
WMS Parameters [name,description,example]
----------------------------------------------------------------
BBOX: lonmin,latmin,lonmax,latmax	-180,0,180,90
COLORSCALERANGE: -4,4  
CRS	EPSG: 4283 
FORMAT:	image/png   
HEIGHT:	256
LAYERS:	netCDF variable name
NUMCOLORBANDS: 254
PALETTE: redblue or a series of matplotlib readable colors (names or hexcodes)
REQUEST:	
    GetMap
    GetLegendGraphic
    GetFeatureInfo
    (GetFullFigure)
SERVICE: WMS
STYLES:	boxfill/redblue
STYLE:
    grid
    contour
TIME: 2010-09-20T00:00:00 (assume ISO8601 format)
TIMEINDEX: 0
TRANSPARENT: true
WIDTH: 256

-----------------
Custom parameters 
-----------------
DAP_URL - location of netcdf file (not full string of dap data request - we don't
parse the DAP output, we use pydap.)

The REQUEST parameter allows the GetFigure option, which plots a full figure,
that is, the requested variable, a colour bar and a title all in one image.
Assumptions are made about the shape of the data: it must be two dimensional.

EXTEND: ['min','max','neither','both']
COLORBOUNDS: boundary at which colours change (intended for use with manual palette) 
CBARTICKS: where ticks are drawn on the colorbar
CBARTICKLABELS: labels for ticks drawn on the colorbar
MAX_COLOR: Color for values above given range
MIN_COLOR: Color for values below given range

------------------
System Requirments
------------------
   - numpy
   - pydap
   - basemap 
   - matplotlib

Instructions:
     Python libraries need to be executable by 'all' (permissions 755). The
     pythonpath on your system, and the location of the python executable,
     may vary.
    
     When running in ipython use ; to seperate variabels
     %run map_plot_form.py variable='hr24_prcp';foo=bar

    For guidance in using the non-pyplot interface:
    http://matplotlib.sourceforge.net/leftwich_tut.txt

    Ensure the cgi-bin directory contains a soft link called python to the 
    python installation that will be used. /usr/bin/env python searches the 
    path for python, which is not appropriate on all systems especially for
    cgi-scripts which will not have a user's path variable, or in the case
    of mulitple installs where the system default will not always have numpy,
    matplotlib because of cgi-bin user

ENVIRONMENT VARIABLES
    PYTHON_PATH_PASAP

Look for something like the following in the Apache config for PASAP:
  SetEnv PYTHON_PATH_PASAP /home/acharles/local/python/2.4.3/lib/python2.4/site-packages
  (on the AIFS machine: /etc/httpd/conf.d/pasap.conf)

TODO:
    Different Projections
    Add urls for dependencies to documentation
    Change dap call to get only the slice of data required.
    Add case insensitivity for parameters
        form_lc = make_lowercase(cgi.FieldStorage)
    Handle the case of a variable not being found at the url gracefully
    Image height/width ratios close to or less than one cause strange errors.
    try/except variable retrieval in case of not in file

An example request:
http://poamaloc/experimental/pasap/cgi-bin/map_plot_wms.py?TRANSPARENT=true&FORMAT=image%2Fpng&LAYERS=hr24_prcp&VERSION=1.3.0&EXCEPTIONS=INIMAGE&SERVICE=WMS&REQUEST=GetMap&STYLES=&CRS=EPSG%3A4283&BBOX=118.94580078125,-48.45361328125,190.79638671875,15.75048828125&WIDTH=817&HEIGHT=730

# Andrew Charles 201012-11
# Roald de Wit
# Kay Shelton
# Elaine Miles
# Andrew Charles 201101-27
# Andrew Charles 20110222 -- moving to pydap 3. Allowing full URL to be passed (i.e. to grab
# a slice from the dap url.)
# Andrew Charles 20130205 -- made grid and contours consistent, cleaned up some things

"""

# Python setup and importation of required modules.
import cgi
import cgitb
cgitb.enable()
import sys
import os
import site
import traceback

# Set up the pythonpath
for path in os.environ.get('PYTHON_PATH_PASAP','').split(':'):
    site.addsitedir(path)

# Set up a temporary directory for matplotlib    
os.environ['MPLCONFIGDIR'] = '/tmp/'
import numpy as np
from pydap.client import open_url
from time import strftime, time, strptime
import datetime
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.basemap import Basemap, addcyclic
try:
    from mpl_toolkits.basemap import date2num, num2date
except ImportError:
    from netCDF4 import date2num, num2date

from mpl_toolkits.basemap import interp
from scipy.interpolate import interpolate
import StringIO
from scipy.ndimage import gaussian_filter

# Create a cache for storing data from urls
cache = {}

# Whether or not to use the cache. Having the cache enabled can cause
# issues when changing files with the same name in place
useCache = False

def application(environ, start_response):
    start = time()

    # Only web servers with the DEBUG env var set will get 
    # to show debugging outpu
    DEBUG = bool(environ.get('DEBUG', False))

    params = cgi.FieldStorage(fp=environ['wsgi.input'], environ=environ)
    download = params.getvalue('DOWNLOAD', False);

    # Catch any possible exception
    try:
        (output, content_type) = doWMS(params)
    except Exception, err:
        output = None
        errorMessage = str(err) + traceback.format_exc() 

    # For performance testing
    #print "Duration: %s" % str(time() - start)
    #print "TYPE:", type(output)
    if output:
        if download:
            filename = "map.png"
            start_response('200 OK', [
                ('Content-Disposition', 'attachment;filename=%s' % filename),
                ('Content-Type', content_type),
                ("Content-length", str(len(output))),
                ])
        else:
            start_response('200 OK', [
                ('Content-Type', content_type),
                ("Content-length", str(len(output))),
                ])

        return [output]
    else:
        error = "<html><head><title>Sorry</title></head><body>" 
        error += "<p>There was an error and the server was unable to process your request.</p>"
        if errorMessage:
            host = environ.get('HTTP_HOST')
            uri = environ.get('REQUEST_URI')
            scriptname = environ.get('SCRIPT_NAME')
            url = "http://%s%s" % (host, uri)
            sys.stderr.write('*** ERROR ***\n')
            sys.stderr.write('* URL: %s\n' % str(url))
            sys.stderr.write('* MESSAGE: %s\n' % str(errorMessage))
            if DEBUG:
                error += "<p><b>Request:</b></p>" 
                error += "<ul>" 
                error += "<li><b>Script</b>: http://%s%s</li>" % (cgi.escape(host), cgi.escape(scriptname))
                error += "</ul>" 
                error += "<p><b>Params:</b></p>" 
                error += "<ul>" 
                for key in params:
                    param = params.getvalue(key)
                    error += "<li><b>%s</b>: %s</li>" % (key, cgi.escape(param))
                error += "</ul>" 
                error += "<p><b>Error message:</b></p><pre>%s</pre>" % cgi.escape(errorMessage)
        error += "</body></html>"
        start_response('500 Internal Error', [
            ('Content-Type', 'text/html'),
            ("Content-length", str(len(error))),
            ], sys.exc_info)
        return [error,]

def doWMS(params):
    """ Provides a wrapper for the map-plot function. """
    varname = params.getvalue('LAYERS', params.getvalue('LAYER','hr24_prcp'))
    bbox = params.getvalue('BBOX','-180,-90,180,90')
    projection_code = params.getvalue('CRS', 'EPSG:4283')
    url = params.getvalue('DAP_URL',
        'http://opendap.bom.gov.au:8080/thredds/dodsC/PASAP/atmos_latest.nc')
    imgwidth = int(params.getvalue('WIDTH',256))
    imgheight = int(params.getvalue('HEIGHT',256))
    format = params.getvalue('FORMAT', 'image/png')
    transparent = params.getvalue('TRANSPARENT', "true")
    request = params.getvalue('REQUEST', 'GetMap')
    time = params.getvalue('TIME','Default')
    palette = params.getvalue('PALETTE','jet')
    style = params.getvalue('STYLE','grid')
    ncolors = int(params.getvalue('NCOLORS',7))
    timeindex = params.getvalue('TIMEINDEX','Default')
    colrange_in = params.getvalue('COLORSCALERANGE','auto')
    cbar_ticks = params.getvalue('CBARTICKS',None)
    cbar_ticklabels = params.getvalue('CBARTICKLABELS',None)
    colorbounds = params.getvalue('COLORBOUNDS',None)
    max_color = params.getvalue('MAXCOLOR',None)
    min_color = params.getvalue('MINCOLOR',None)
    extend = params.getvalue('EXTEND',None)

    # Cater for relative urls
    if url.startswith('/'):
        url = 'http://localhost' + url

    if colorbounds is not None:
        colorbounds = [float(colr) for colr in colorbounds.split(',')]
        # If colorbounds is set then we ignore colorrange
        colorrange = colorbounds[0],colorbounds[len(colorbounds)-1]
        ncolors = len(colorrange) + 1

    else:
        # THREDDS allows for the colorscalerange to be set to auto
        # We do not handle that case currently.
        if colrange_in == 'auto':
            colrange_in = '-4,4'
        colorrange = tuple([float(a) for a in colrange_in.rsplit(',')])

    #if cbar_ticks_in == 'Default':
    #    interval = float(colorrange[1] - colorrange[0]) / float(ncolors) 
    #    cbar_ticks =  np.arange(colorrange[0], colorrange[1] + interval*.1, interval).round(1)
    #    cbar_ticks[cbar_ticks==-0]=0
    #    cbar_ticks =  ",".join(["%s" % el for el in list(cbar_ticks)])

    save_local_img = bool(int(params.getvalue('SAVE_LOCAL',0)))

    #Turn transparent param into a boolean
    transparent = transparent.lower() == "true"

    # Allow for multiple output formats. Defaults to png
    valid_formats = ["png", "png8"]
    format = format.lower().split("/")[-1] # "image/png" -> png, "png" -> "png"
    if format not in valid_formats:
        format = "png"

    if request.lower() != "getfeatureinfo":
        # Might be nicer to just pass a dict to mapdap()
        output = mapdap(varname=varname,bbox=bbox,url=url,imgheight=imgheight,imgwidth=imgwidth,
            request=request,time=time,timeindex=timeindex,save_local_img=save_local_img,
            colorrange=colorrange,palette=palette,style=style,ncolors=ncolors,format=format,transparent=transparent,cbar_ticks=cbar_ticks,cbar_ticklabels=cbar_ticklabels,colorbounds=colorbounds,extend=extend,max_color=max_color,min_color=min_color)

        return (output, "image/png")
    else:
        # Initial work on GetFeatureInfo
        output = '<?xml version="1.0" encoding="UTF-8"?><wfs:FeatureCollection xmlns="http://www.opengis.net/wfs" xmlns:wfs="http://www.opengis.net/wfs" xmlns:opengeo="http://opengeo.org" xmlns:gml="http://www.opengis.net/gml" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://opengeo.org http://demo.opengeo.org:80/geoserver/wfs?service=WFS&amp;version=1.0.0&amp;request=DescribeFeatureType&amp;typeName=opengeo:roads http://www.opengis.net/wfs http://demo.opengeo.org:80/geoserver/schemas/wfs/1.0.0/WFS-basic.xsd"><gml:boundedBy><gml:Box srsName="http://www.opengis.net/gml/srs/epsg.xml#26713"><gml:coordinates xmlns:gml="http://www.opengis.net/gml" decimal="." cs="," ts=" ">591943.9375,4925605 593045.625,4925845</gml:coordinates></gml:Box></gml:boundedBy><gml:featureMember><opengeo:roads fid="roads.90"><opengeo:variable>%s</opengeo:variable><opengeo:value>%s</opengeo:value><opengeo:the_geom><gml:Point srsName="http://www.opengis.net/gml/srs/epsg.xml#26713"><gml:coordinates>190,-30</gml:coordinates></gml:Point></opengeo:the_geom></opengeo:roads></gml:featureMember></wfs:FeatureCollection>'

        output = output % (varname, "dummy")
        return (output, "text/plain") 

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.
    
    cmap: colormap instance, eg. cm.jet. 
    N: Number of colors.
    
    Example
    x = resize(arange(100), (5,100))
    djet = cmap_discretize(cm.jet, 5)
    imshow(x, cmap=djet)
    
    from http://www.scipy.org/Cookbook/Matplotlib/ColormapTransformations
    """
    cdict = cmap._segmentdata.copy()
    # N colors
    colors_i = np.linspace(0,1.,N)
    # N+1 indices
    indices = np.linspace(0,1.,N+1)
    for key in ('red','green','blue'):
                # Find the N colors
        D = np.array(cdict[key])
        I = interpolate.interp1d(D[:,0], D[:,1])
        colors = I(colors_i)
        # Place these colors at the correct indices.
        A = np.zeros((N+1,3), float)
        A[:,0] = indices
        A[1:,1] = colors
        A[:-1,2] = colors
        # Create a tuple for the dictionary.
        L = []
        for l in A:
            L.append(tuple(l))
        cdict[key] = tuple(L)
    # Return colormap object.
    return mpl.colors.LinearSegmentedColormap('colormap',cdict,1024)

def read_string_var(dataset,varname):
    """ Read a string variable via OPEnDAP.
        The means for doing this varies between implementations.
        This function is not extensively tested, and may only
        work for THREDDS.
    """
    var = dataset[varname]
    dims = list(var.dimensions)
    names = []
    if 'slen' in dims:
        # Pydap style response
        dimdex = dims.index('slen')
        for i in range(dataset[varname].array.shape[0]):
            names.append(''.join(dataset[varname].array[i] ))
    else:
        # THREDDS style response
        names = dataset[varname]
    return(np.array(names))

def cellbounds(ar):
    """ 
        Generate bounds from midpoints.
        ar -- a coordinate array in one dimension

        Returns lower/upper bounds of grid points defined at centres ar
        Based on code in CDAT. 
    """
    
    leftPoint = np.array([1.5*ar[0]-0.5*ar[1]])
    midArray = (ar[0:-1]+ar[1:])/2.0
    rightPoint = np.array([1.5*ar[-1]-0.5*ar[-2]])
    bnds = np.concatenate((leftPoint,midArray,rightPoint))
    return bnds

def transform_lons(coords,lon,f):
    """ Take bounding box longitudes and transform them so that basemap plots sensibly. """
    """
    Arguments
    coords -- a tuple compose of lonmin,lonmax
    lon -- numpy array of longitudes
    f -- numpy array of the field being plotted

    >>> trans_coords()

    This logic can probably be simplified as it was built incrementally to solve several
    display issues. See tests/allplots.shtml for the tests that drove this function.

    """
    x1,x2 = coords
    lont = lon
    
    # To handle 360 degree plots
    if x2 == x1:
        x2 = x1 + 360
   
    # Basemap doesn't always play well with negative longitudes so convert to 0-360
    lon2360 = lambda x: ((x + 360.) % 360.)
    if x2 < 0:
        x2 = lon2360(x2)
        x1 = lon2360(x1)
        lont = lon2360(lont)

    # If this has resulted in xmin greater than xmax, need to reorder
    if x2 < x1:
        x2 = x2 + 360
    
    # If the start lon is less than zero, then convert to -180:180
    # It's not clear this will ever be executed, given the above code
    if (x1 < 0) or (x2 == x1 and x2 == 180):
        x1 = ((x1 + 180.) % 360.) - 180.
        x2 = ((x2 + 180.) % 360.) - 180.
        lont = ((lont + 180.) % 360.) - 180.

    # If this has resulted in xmin greater than xmax, we need to reorder
    if x2 < x1:
        x2 = x2 + 360

    # If the x2 range is greater than 360 (plots that span both dl and pm)
    # then remap the longitudes to this range
    if x2 > 360:
       idx = lont < x1
       lont[idx] = lont[idx] + 360
   
    # Remap the longitudes for this case too
    if x1 < 0 and x2 > 180:
       idx = lont < x1
       lont[idx] = lont[idx] + 360
   
    # The special case of 0-360
    if x2 == x1:
        if x2 == 0:
            x1 = 0
            x2 = 360
        else:
            x2 = abs(x2)
            x1 = -x2
    
    coords = x1,x2
    # Ensure lons are ascending, and shuffle the field with the same indices
    idx = np.argsort(lont)
    lont = lont[idx]
    ft = f[:,idx]
    ft, lont = addcyclic(ft,lont)
    return coords,lont,ft

def figurePlotDims(imgheight,imgwidth,coords,plot_max_xfrac=0.7,plot_max_yfrac=0.7):
    """
     Compute a new x and y fraction such that the lat/lon aspect ratio is constant.
        
     imgwidth,imgheight, 
     coords -- lonmin,latmin,lonmax,latmax,
     the maximum fraction to be taken up by the plot, 

    """
    lonmin,latmin,lonmax,latmax = coords
    plot_max_height=plot_max_yfrac*imgheight
    plot_max_width=plot_max_xfrac*imgwidth
    nlat = float(latmax-latmin)
    nlon = float(lonmax-lonmin)
    plot_aspect_ratio = plot_max_width/plot_max_height
    latlon_aspect_ratio = (nlon/nlat) * (plot_aspect_ratio)
    desired_aspect_ratio = 1.0
    
    if latlon_aspect_ratio > desired_aspect_ratio:
        # Image is needs to be narrower
        plot_xfrac = (desired_aspect_ratio) * (nlon/nlat) * (plot_max_yfrac) \
            * (float(imgheight)/imgwidth)
        plot_yfrac = plot_max_xfrac
    else:
        # Image is needs to be shorter
        plot_yfrac = (1./desired_aspect_ratio) * (nlat/nlon) * (plot_max_width) \
            * (float(imgwidth)/imgheight)
        plot_xfrac = plot_max_xfrac

    # Ensure we are within the bounds!
    if (plot_yfrac > plot_max_yfrac):
        plot_xfrac = plot_xfrac * plot_max_yfrac/plot_yfrac
        plot_yfrac = plot_max_yfrac
    if (plot_xfrac > plot_max_xfrac):
        plot_yfrac = plot_yfrac * plot_max_xfrac/plot_xfrac
        plot_xfrac = plot_max_xfrac

    return (0.08,0.08,plot_xfrac,plot_yfrac)

def get_pasap_plot_title(dset,varname = 'hr24_prcp',timestep= 0):
    """ Given an open pydap object (dset), and some extra information, return a nice
        plot title.
    """
    header = "PASAP: Dynamical Seasonal Outlooks for the Pacific."
    subheader1 = "Outlook based on POAMA 2 CGCM adjusted for historical skill"
    subheader2 = "Experimental outlook for demonstration and research only"

    if 'time' in dset.keys() and 'init_date' in dset.keys():
        time_var = dset['time']
        if 'units' in time_var.attributes.keys():
            time_units = time_var.attributes['units']
        else:
            time_units = ''
        valid_time = datetime.datetime.strftime(
            num2date(time_var[timestep],time_units),"%Y%m%d")
        start_date = datetime.datetime.strftime(
            num2date(dset['init_date'][0],time_units),"%Y%m%d")
    else:
        start_date = 'unknown' 

    if 'units' in dset[varname].attributes.keys():
        units = dset[varname].attributes['units']
    else:
        units = 'no units provided'

    if 'time_label' in dset.keys():
        period_label = str(read_string_var(dset,'time_label')[timestep])
    else:
        period_label = 'unknown'
    titlestring = header + '\n' \
                  + subheader1 + '\n'  \
                  + subheader2 + '\n'  \
                  + "Variable: " + varname + ' (' + units + ')' + '\n' \
                  + 'Model initialised: ' + start_date + '\n' \
                  + 'Forecast period: ' + period_label 

    return titlestring

def get_plot_title(dset,varname='hr24_prcp',timestep=0):
    """ Given an open pydap object, and some extra information, return a nice
        plot title.
        TODO: This should return as generic a title as possible based only
        on the dataset metadata with no further assumptions.
    """

    if 'time' in dset.keys() and 'init_date' in dset.keys():
        time_var = dset['time']
        if 'units' in time_var.attributes.keys():
            time_units = time_var.attributes['units']
        else:
            time_units = ''
        valid_time = datetime.datetime.strftime(
            num2date(time_var[timestep],time_units),"%Y%m%d")
        start_date = datetime.datetime.strftime(
            num2date(dset['init_date'][0],time_units),"%Y%m%d")
    else:
        start_date = 'unknown' 

    if 'units' in dset[varname].attributes.keys():
        units = dset[varname].attributes['units']
    else:
        units = ''#no units provided'

    if 'time_label' in dset.keys():
        period_label = str(read_string_var(dset,'time_label')[timestep])
        period_label = period_label.replace("\n", ", ")
    else:
        period_label = ''

    if 'lead_time' in dset.keys():
        lead_time = str(read_string_var(dset,'lead_time')[timestep])
        lead_time_label = ', Lead time: ' + lead_time + ' months'
        if lead_time == '1':
            lead_time_label = ', Lead time: ' + lead_time + ' month'
    else:
        lead_time_label = ""

    header = "PACCSAP: Dynamical Seasonal Outlooks for the Pacific.\n"
    subheader1 = "Outlook based on POAMA 2 CGCM adjusted for historical skill.\n"
    subheader2 = "Experimental outlook for demonstration and research only.\n"
    titlestring = header + subheader1 +  subheader2 + 'Variable:  ' + varname + '(' + units + ')' + '\n' \
                  + 'Model initialised: ' + start_date + '\n' + 'Forecast period: ' + period_label \
                  + lead_time_label

    return titlestring


def get_colmap(palette,ncolors,colorbounds=None,colorrange=None,
    max_color=None,min_color=None):
    """ Get a colormap and normalise instance.

        palette can be a string list of colors, or the name of a palette
    # if a colormap could not be found, we assume that palette is a string separating colors
    # like this: "#996518,#FE0000,#FE6532,#F99634,#FEFE33,#F2F2F2,#FEFEFE,#FEFEFE,#D5D5D5,#CBFECB,#65CB65,#6699FE,#0065FE,#660099"
    if not colormap:

    """
    colormap = mpl.cm.get_cmap(palette,ncolors)
    norm = None

    if not colormap:
        colormap = mpl.colors.LinearSegmentedColormap.from_list(
            'custom', palette.split(',')[0:ncolors])
        
        if colorbounds is not None:
            norm = mpl.colors.BoundaryNorm(colorbounds,ncolors=256,clip=False)
        elif colorrange is not None:
            norm = mpl.colors.Normalize(vmin=colorrange[0], vmax=colorrange[1], clip=False)

        # At this point, norm has been set, so the last few clauses in the function
        # will not execute.
   
    if max_color is not None: 
        colormap.set_over(max_color)

    if min_color is not None: 
        colormap.set_under(min_color)
    
    if norm is None and colorbounds is not None:
        norm = mpl.colors.BoundaryNorm(colorbounds,ncolors=ncolors,clip=False) 
    elif (norm is None) and (colorbounds is None) and (colorrange is not None):
        norm = mpl.colors.Normalize(vmin=colorrange[0], vmax=colorrange[1], clip=False)

    return colormap,norm


def mapdap(
    varname = 'hr24_prcp',
    bbox = '-180,-90,180,90',
    url = 'http://opendap.bom.gov.au:8080/thredds/dodsC/PASAP/atmos_latest.nc',
    timeindex = 'Default',
    imgwidth = 256,
    imgheight = 256,
    request = 'GetMap',
    format = 'png',
    transparent = True,
    time = 'Default',
    save_local_img = False,
    colorrange = None,#(-4,4),
    palette = 'RdYlGn',
    colorbounds = None,
    style = 'grid',
    ncolors = None,
    mask = -999,
    plot_mask = True,
    mask_varname = 'mask',
    mask_value = 1.0,
    cbar_ticks = None,
    cbar_ticklabels = None,
    extend = None,
    max_color = None,
    min_color = None
    ):
    """ Using Basemap, create a contour plot using some dap available data 
   
        Data is assumed to have dimensions [time,lat,lon] 
            TODO -- deal with other shapes
            TODO -- determine the dimension ordering using CF convention

        varname -- name of variable in opendap file
        bbox -- lonmin,latmin,lonmax,latmax for plot
        url -- OPEnDAP url
        timeindex -- time index to plot
        imgwidth,imgheight -- size of png image to return
        request -- 'GetMap','GetLegend','GetFullFigure'
        time -- time vale to plot. Assumes a particular format."%Y-%m-%dT%H:%M:%S"
        mask -- mask out these values
        if plot_mask is True, mask_varname and mask_value must be given
    
    """
    lonmin,latmin,lonmax,latmax = tuple([float(a) for a in bbox.rsplit(',')])

    # PART 1: Find the opendap data source and download the variable's data
 
    # It's not clear there is any point in the cache. Pydap doesn't actually
    # download data until you subscript 
    if useCache:
        if url not in cache:
            dset = open_url(url)
        else:
            dset = cache[url]
    else:
        dset = open_url(url)

    timestep=0
    if 'time' in dset[varname].dimensions and 'time' in dset:
        # Get the correct time.
        time_var = dset['time']
        time_units = time_var.attributes['units']
        available_times = np.array(time_var[:])

        # TODO there is a potential conflict here between time and timeindex.
        # On the one hand we want to allow using the actual time value.
        # On the other hand we want to make it easy to get a time index
        # without knowing the value.
        if timeindex == 'Default':
            timestep=0
        else:
            timestep=int(timeindex)
        if time != 'Default':
            dtime = datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S" )
            reftime = date2num(dtime,time_units)
            timestep = np.where(available_times >= reftime)[0].min()

        # Determine what index is time
        # TODO Obviously it could be 1,2,3 even 4.
        if dset[varname].dimensions[2] == 'time':
            var = dset[varname][:,:,timestep]
        else:
            var = dset[varname][timestep,:,:]
    else:
        var = dset[varname][:,:]

    # TODO Get only the section of the field we need to plot (if efficient)
    # TODO Determine lat/lon box indices and only download this slice
    # Needs more thought - the idea here is to only grab a slice of the data
    # Need to grab a slightly larger slice of data so that tiling works.
    #lat_idx = (lat > latmin) & (lat < latmax)
    #lon_idx = (lon > lonmin) & (lon < lonmax)
    #lat = dset['lat'][lat_idx]
    #lon = dset['lon'][lon_idx]
    #latdx1 = np.where(lat_idx)[0].min()
    #latdx2 = np.where(lat_idx)[0].max()
    #londx1 = np.where(lon_idx)[0].min()
    #londx2 = np.where(lon_idx)[0].max()
    #var = var[latdx1:latdx2+1,londx1:londx2+1]
    #var = dset[varname][timestep,latdx1:latdx2+1,londx1:londx2+1]
    # TODO Set default range (the below does not work)
    
    if 'lat' in var.dimensions:
        lat = dset['lat'][:]
        lon = dset['lon'][:]
    else:
        lat = dset['latitude'][:]
        lon = dset['longitude'][:]
 
    # Create an array from the pydap returned type. (?)
    vardata = var[varname][:,:]

    """ Process the masking rules. Preferred approach is to use the 'missing_value' attrib. """

    # First mask any infinities, extremely large values, and NaNs
    mask_arr = np.isinf(vardata)
    mask_arr = mask_arr | np.isnan(vardata)
    #mask_arr = mask_arr | (vardata == -999)
    mask_arr = mask_arr | (vardata >= 1e30)
   
    # Then look to see if a mask variable is defined in the file 
    # TODO Check that it's a bool!
    # TODO Deprecate this method (requires adjusting PASAP data outputs)
    if plot_mask:
        if 'mask' in dset.keys():
            if 'time' in dset:
                maskvar = dset['mask'][timestep,:,:]
            else:
                maskvar = dset['mask'][:,:]

            mask_arr = mask_arr | maskvar
       
    # Finally look for the missing value attribute
    try:
        var_missing_val = var.attributes['missing_value']
    except KeyError:
       var_missing_val = ''
    mask_arr = mask_arr | (vardata == var_missing_val)
        
    varm = np.ma.masked_array(vardata,mask=mask_arr)

    xcoords = lonmin,lonmax
    # Call the trans_coords function to ensure that basemap is asked to
    # plot something sensible.
    xcoords,lon,varm = transform_lons(xcoords,lon,varm)
    lonmin,lonmax = xcoords
    varnc = dset[varname]

    # TODO: More special cases could be added here.
    try:
        var_units = varnc.attributes['units']
        if var_units=='deg C':
            var_units = u"\u00b0C"
    except KeyError:
       var_units = '' 

    # END PART 1. The data has been obtained

    # PART 2. Set up the Basemap object

    # Note:
    # For the basemap drawing we can't go outside the range of coordinates
    # WMS requires us to give an empty (transparent) image for these spurious lats
    # Basemap terminorlogy: uc = upper corner, lc = lower corner
    bmapuclon=lonmax
    bmaplclon=lonmin
    bmapuclat=min(90,latmax)
    bmaplclat=max(-90,latmin)
    if bmaplclat==90:
        bmaplclat = 89.0
    if bmapuclat==-90:
        bmapuclat = -89.0

    # TODO set figsize etc here  
    fig = mpl.figure.Figure()
    canvas = FigureCanvas(fig)
    
    ax = fig.add_axes((0,0,1,1),frameon=False,axisbg='k',alpha=0,visible=False)
    m = Basemap(projection='cyl',resolution='c',urcrnrlon=bmapuclon,
        urcrnrlat=bmapuclat,llcrnrlon=bmaplclon,llcrnrlat=bmaplclat,
        suppress_ticks=True,fix_aspect=False,ax=ax)

    DPI=100.0

    # Convert the latitude extents to Basemap coordinates
    bmaplatmin,bmaplonmin = m(latmin,lonmin)
    bmaplatmax,bmaplonmax = m(latmax,lonmax)
    lon_offset1 = abs(bmaplclon - bmaplonmin)
    lat_offset1 = abs(bmaplclat - bmaplatmin)
    lon_offset2 = abs(bmapuclon - bmaplonmax)
    lat_offset2 = abs(bmapuclat - bmaplatmax)
    lon_normstart = lon_offset1 / abs(bmaplonmax - bmaplonmin)
    lat_normstart = lat_offset1 / abs(bmaplatmax - bmaplatmin)
    ax_xfrac = abs(bmapuclon - bmaplclon)/abs(bmaplonmax - bmaplonmin)
    ax_yfrac = abs(bmapuclat - bmaplclat)/abs(bmaplatmax - bmaplatmin)

    # Set plot_coords, the plot boundaries. If this is a regular WMS request,
    # the plot must fill the figure, with whitespace for invalid regions.
    # If it's a full figure, we need to make sure there is space for the legend
    # and also for the text.
    if request == 'GetFullFigure':
        coords = lonmin,latmin,lonmax,latmax
        plot_coords = figurePlotDims(imgheight,imgwidth,coords)
        # New line to fix downloaded map
        # Want islands to appear when map is downloaded
        # resolution c, l, i, h, f
        m = Basemap(projection='cyl',resolution='i',area_thresh = 1,urcrnrlon=bmapuclon,
                    urcrnrlat=bmapuclat,llcrnrlon=bmaplclon,llcrnrlat=bmaplclat,
                    suppress_ticks=True,fix_aspect=False,ax=ax)
    else:
        plot_coords = (lon_normstart,lat_normstart,ax_xfrac,ax_yfrac)
    # Commented out, appears to be repeating above statement
    #    m = Basemap(projection='cyl',resolution='c',urcrnrlon=bmapuclon,
    #        urcrnrlat=bmapuclat,llcrnrlon=bmaplclon,llcrnrlat=bmaplclat,
    #        suppress_ticks=True,fix_aspect=False,ax=ax)

    # DONE setting up the Basemap object

    ax = fig.add_axes(plot_coords,frameon=False,axisbg='k')

    m.ax = ax
    #varm,lonwrap = addcyclic(varm,lon)
    varm,lonwrap = varm,lon
    
    # PART 3: TRICKY COLORBAR LOGIC

    if extend is None:
        extend = 'both'

    if (colorrange == 'auto') or (colorrange is None):
        colorrange = np.min(var),np.max(var)

    # TODO: delete once tested as this is now impotent
    if ((extend == 'max') or (extend == 'min')): 
        if (colorbounds is None):
            ncolors = ncolors# - 1
    
    if colorrange is None and colorbounds is not None:
        # This is what to do if you get explicit thresholds
        colorrange = colorbounds[0],colorbounds[-1]
        if ncolors is None:
            ncolors = len(colorbounds) + 1

    if colorbounds is None and colorrange is not None:
        increment = float(colorrange[1]-colorrange[0]) / float(ncolors) 
        colorbounds = list(np.arange(colorrange[0],colorrange[1]+increment/2.,increment))

    colormap,norm = get_colmap(palette,ncolors,colorbounds=colorbounds,colorrange=colorrange,
        max_color=max_color,min_color=min_color)

    if cbar_ticks in ["auto", None]:
        cbar_ticks = colorbounds
    else:
        cbar_ticks = [float(ct) for ct in cbar_ticks.split(',')]

    if cbar_ticklabels is not None:
        cbar_ticklabels = cbar_ticklabels.split(',')

    if style == 'contour':
        # Interpolate to a finer resolution
        # TODO: make this sensitive to the chosen domain

        lat_idx = np.argsort(lat)
        lat = lat[lat_idx]
        varm = varm[lat_idx,:]
        mask_arr = varm.mask[lat_idx,:]
        varm = np.ma.masked_array(varm,mask=mask_arr)

        data_lonmin = min(lonwrap)
        data_lonmax = max(lonwrap)
        data_latmin = min(lat)
        data_latmax = max(lat)

        new_lons = np.arange(data_lonmin-1.0,data_lonmax+1.0,1.0)
        new_lats = np.arange(data_latmin-1.0,data_latmax+1.0,1.0)
        newx,newy = m(*np.meshgrid(new_lons[:],new_lats[:]))
        oldx,oldy = m(*np.meshgrid(lonwrap[:],lat[:]))
        x = newx
        y = newy

        """
        Interpolation and Smoothing
        ---------------------------
        We need to interpolate the data to a finder grid (this helps with generating
        less jagged contours), and apply a small amount of smoothing (this helps to 
        avoid showing model grid boxes.
 
        The existence of a mask complicates this.

        We should be able to trust the data's mask variable. But because no physical variables 
        we plot will be less than -900 we extend the mask here to cover this.
        
        When interpolating to the finer grid a two pass interpolation to deal
        with the mask is used. The first pass does a bilinear, the next pass does a
        nearest neighbour to ensure points that were masked in the orginal data are
        masked in the interpolated data.

        """

        maskdata = mask_arr
        varm_masked = np.ma.masked_array(varm,mask=maskdata)
        varm_nomask = varm.copy()

        # This code from Stack Overflow, effectively does a nearest neighbour interpolation
        # to infill the masked regions to avoid the gaussian filter from dragging down
        # the values of points near the mask.
        xss = np.arange(-4,4)
        yss = np.arange(-4,4)
        neighbours = []
        
        for xs in xss:
            for ys in yss:
                neighbours.append((xs,ys)) 
        for hor_shift,vert_shift in neighbours:
            a_shifted = np.roll(varm_nomask,shift=hor_shift,axis=1)
            a_shifted = np.roll(a_shifted,shift=vert_shift,axis=0)
            idx = ~a_shifted.mask * varm_masked.mask
            varm_nomask[idx] = a_shifted[idx]
        
        varm_bl = interp(varm_nomask, lonwrap[:], lat[:], newx, newy,order=1)
        varm_nn = interp(varm_masked, lonwrap[:], lat[:], newx, newy,order=0)

        # Apply a modest blurring after interpolation to smooth out the
        # contour lines.
        varm_gf = gaussian_filter(varm_bl,1.0,mode='wrap')
        varm = np.ma.masked_array(varm_gf,mask=varm_nn.mask)
        varm[varm_nn.mask == True] = varm_nn[varm_nn.mask == True]
        # INTERPOLATION FINISHED

        colmap,norm = get_colmap(palette,ncolors,colorbounds=colorbounds,colorrange=colorrange,
            max_color=max_color,min_color=min_color)
        
        x,y = m(*np.meshgrid(new_lons[:],new_lats[:]))

        main_render = m.contourf(x,y,varm[:,:],colorbounds,extend=extend,
            cmap=colormap,norm=norm,ax=ax)

        contours = m.contour(x,y,varm,colorbounds,colors='k',ax=ax,linewidths=0.5)
        contours.clabel(colors='k',rightside_up=True,fmt='%1.1f',inline=True,fontsize=8)

    elif style == 'grid':
        # pcolor thinks it is getting cell edges, not cell centres,
        # but model data is stored by cell centres
        # here we calculate the cell edges
        cbx = cellbounds(lonwrap[:])
        cby = cellbounds(lat[:])
        x,y = m(*np.meshgrid(cbx,cby))
        mask = varm.mask[:,:]

        colmap,norm = get_colmap(palette,ncolors,colorbounds=colorbounds,colorrange=colorrange,
            max_color=max_color,min_color=min_color)

        main_render = m.pcolormesh(x,y,varm[:,:],shading='flat',norm=norm,
            vmin=colorrange[0],vmax=colorrange[1],cmap=colormap,ax=ax)

    elif style == 'grid_threshold':
        cbx = cellbounds(lonwrap[:])
        cby = cellbounds(lat[:])
        x,y = m(*np.meshgrid(cbx,cby))
        main_render = m.pcolormesh(x,y,varm[:,:],norm=norm,shading='flat',
            vmin=colorrange[0],vmax=colorrange[1],cmap=colormap,ax=ax)

    else:
        cbx = cellbounds(lonwrap[:])
        cby = cellbounds(lat[:])
        x,y = m(*np.meshgrid(cbx,cby))
        main_render = m.pcolormesh(x,y,varm[:,:],norm=norm,shading='flat',
            vmin=colorrange[0],vmax=colorrange[1],cmap=colormap,ax=ax)
        

    fig.set_dpi(DPI)
    fig.set_size_inches(imgwidth/DPI,imgheight/DPI)

    title_font_size = 9
    tick_font_size = 8
    if request == 'GetFullFigure':
        # Default - draw 5 meridians and 5 parallels
        #n_merid = 5 # this gets worked out dynamically so that lon spacing = lat spacing
        n_para = 5

        pint = int((latmax - latmin)/float(n_para))
        if pint < 1:
            pint = 1 #minimum spacing is 1 degree
        base = pint
        parallels = [latmin + i*pint for i in range(0,n_para+1)] 
        parallels = [ int(base * round( para / base)) for para in parallels]

        mint = pint
        n_merid = int((lonmax - lonmin)/float(mint))
        meridians = [lonmin + i*mint for i in range(0,n_merid+1)]
        meridians = [ int(base * round( merid / base)) for merid in meridians]
        
        m.drawcoastlines(ax=ax)
        
        m.drawmeridians(meridians,labels=[0,1,0,1],fmt='%3.1f',fontsize=tick_font_size)
        m.drawparallels(parallels,labels=[1,0,0,0],fmt='%3.1f',fontsize=tick_font_size)
        m.drawparallels([0],linewidth=1,dashes=[1,0],labels=[0,1,1,1],fontsize=tick_font_size)
        titlex,titley = (0.05,0.98)
        #title = get_pasap_plot_title(dset,varname=varname,timestep=timestep)
        title = get_plot_title(dset,varname=varname,timestep=timestep)
        fig.text(titlex,titley,title,va='top',fontsize=title_font_size)
   
    colorbar_font_size = 8

    # Determine colorbar text formatting
    if colorrange[1] > 100:
        cbar_format = '%4.f'
    else:
        cbar_format = '%4.1f'

    if request == 'GetLegendGraphic':
        # Currently we make the plot, and then if the legend is asked for
        # we use the plot as the basis for the legend. This is not optimal.
        # Instead we should be making the legend manually. However we need
        # to set up more variables, and ensure there is a sensible min and max.

        # See the plot_custom_colors code above
        fig = mpl.figure.Figure(figsize=(64/DPI,256/DPI))
        canvas = FigureCanvas(fig)
        # make some axes
        cax = fig.add_axes([0,0.1,0.2,0.8],axisbg='k')
        # put a legend in the axes

        if style == 'contour':

            cbar = fig.colorbar(mappable=main_render,extend=extend,ticks=cbar_ticks,
                    format=cbar_format,
                    cax=cax,cmap=colormap,boundaries=colorbounds,spacing='proportional')

        elif style == 'grid':

            # Spacing=proportional does not work. dunno why
            cbar = fig.colorbar(mappable=main_render,extend=extend,ticks=cbar_ticks,
                    format=cbar_format,
                    cax=cax,cmap=colormap,norm=norm)

        if cbar_ticklabels is not None:
            cbar.ax.set_yticklabels(cbar_ticklabels)
        
        cbar.set_label(var_units,fontsize=colorbar_font_size)
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(colorbar_font_size)

    elif request == 'GetFullFigure':
        # Add the legend to the figure itself.
        # Figure layout parameters
        # plot_coords = tuple with (xi,yi,dx,dy)
        # legend_coords = tuple with (xi,yi,dx,dy) as per mpl convention
        # First change the plot coordinates so that they do not cover the whole image
        legend_coords = (0.8,0.1,0.02,plot_coords[3])
        cax = fig.add_axes(legend_coords,axisbg='k')

        if style == 'contour':
            cbar = fig.colorbar(mappable=main_render,extend=extend,ticks=cbar_ticks,
                        format=cbar_format,
                        cax=cax,cmap=colormap,
                        spacing='proportional')

        else:
            cbar = fig.colorbar(mappable=main_render,extend=extend,ticks=cbar_ticks,
                    format=cbar_format,
                    spacing='proportional',
                    cax=cax,cmap=colormap)

        if cbar_ticklabels is not None:
            cbar.ax.set_yticklabels(cbar_ticklabels)

        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(colorbar_font_size)
        cbar.set_label(var_units,fontsize=colorbar_font_size)
        
        #Force transparency. TODO: remove forcing
        transparent=False

    #############################################################################
    # Write the img to disk
    import tempfile
    import subprocess
    img_file_name = tempfile.mkstemp(prefix="map_plot_wms_img_", suffix="." + format)[1]
    img_file_handle = open(img_file_name, 'w')
    
    #For ImageMagick to work fine with Matplotlib on www3, we need to remove transparency
    #We bring it back in again by using convert: -transparent white (anything white will
    #become transparent. This may not be the best at all times though.
    if format == "png8":
        fig.savefig(img_file_handle,format='png',transparent=False)
    else:
        fig.savefig(img_file_handle,format=format,transparent=transparent)

    img_file_handle.close()

    # Convert to PNG8 if requested
    if format == "png8":
        if transparent:
            #IMAGEMAGICK with all whites turned transparent 
            subprocess.call(["convert", "-transparent", "white", img_file_name, "png8:" + img_file_name])
        else:
            #IMAGEMAGICK default 
            subprocess.call(["convert", img_file_name, "png8:" + img_file_name])

    value = file(img_file_name,"rb").read()
    
    if save_local_img:
        fig.savefig('map_plot_wms_output.' + format,format=format)
        return

    if useCache and url not in cache:
        cache[url] = dset

    fig = None

    # Remove temp image file
    os.remove(img_file_name)

    return value


threddslocal = "http://localhost:8080/thredds/dodsC/PASAP"

def ocean_mask_test():
        params = cgi.FieldStorage()
        for name, value in {
            "INVOCATION" : "terminal",
            "SAVE_LOCAL": "1",
            "REQUEST" : "GetFullFigure",
            "BBOX" : "100,-50,180,0",
            "WIDTH" : "640",
            "HEIGHT" : "300",
            "DAP_URL" : threddslocal + '/ocean_latest.nc',
            "LAYER" : 'SST',
            "STYLE" : 'grid',
            #"STYLE" : 'contour',
            "COLORSCALERANGE" : '-10,30',
            "NCOLORS" : '12'
            #"STYLE" : 'grid'
        }.items():
            params.list.append(cgi.MiniFieldStorage(name, value))
        doWMS(params)

def atmos_mask_test():
        params = cgi.FieldStorage()
        for name, value in {
            "INVOCATION" : "terminal",
            "SAVE_LOCAL": "1",
            "REQUEST" : "GetFullFigure",
            "BBOX" : "70,-50,180,-5",
            "WIDTH" : "640",
            "HEIGHT" : "300",
            #"DAP_URL" : 'http://opendap.bom.gov.au:8080/thredds/dodsC/PASAP/atmos_latest.nc',
            "DAP_URL" : threddslocal + '/iov_atmos_em_latest.nc',
            "LAYER" : 'hr24_prcpa_nrmse',
            "STYLE" : 'contour',
            "NCOLORS" : '12',
            "COLORSCALERANGE" : '-1.0,1.0'
            #"STYLE" : 'grid'
        }.items():
            params.list.append(cgi.MiniFieldStorage(name, value))
        doWMS(params)


if __name__ == '__main__':
    if 'TERM' in os.environ:
        print "Running from terminal"
        os.environ['INVOCATION'] = 'terminal'

        #print sys.path
        #Some test parameters for debugging
        ocean_mask_test()
        atmos_mask_test()

    elif 'CGI' in os.environ.get('GATEWAY_INTERFACE',''):
        os.environ['INVOCATION'] = 'cgi'
        import wsgiref.handlers
        wsgiref.handlers.CGIHandler().run(application)
    else:
        os.environ['INVOCATION'] = 'wsgi'
        pass

