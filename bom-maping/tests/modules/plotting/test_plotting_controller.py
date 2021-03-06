"""
Unit tests for the PlottingController Class

Author: Stefan Fuchs (s3260968@student.rmit.edu.au)
"""
import unittest
import modules.plotting.plotting_controller as pc
import util.exceptions as ex


class TestPlottingController(unittest.TestCase):
   
    def setUp(self):
        self.param = { "bbox" : {  "min_lat" : "-90.0",
                                   "min_lon" : "0.0",
                                   "max_lat" : "90.0",
                                   "max_lon" : "360.0" 
                                } ,
                        "layers" : ["hr24_prcpa", ] ,
                        "styles" : ["grid", ] ,
                        "crs" : {   "name" : "EPSG" ,
                                    "identifier" : "4283" 
                                } ,
                        "width" : "300" ,
                        "height" : "400" ,
                        "format" : "png" ,
                        "time" : "Default" ,
                        "time_index" : "Default" ,
                        "source_url" : "http://localhost:8001/atmos_latest.nc",
                        "color_scale_range" : ["-10", "10", ] ,
                        "n_colors" : ["10", ] ,
                        "palette" : "jet"
                     }
                    
        self.defaults = { "request":"GetMap",
                          "version":"0.0.1",
                          "bbox" : {  "min_lat" : "-90.0",
                                      "min_lon" : "-180.0",
                                      "max_lat" : "90.0",
                                      "max_lon" : "180.0" 
                                } ,
                          "width" : "256",
                          "height" : "256",
                          "layers" : "hr24_prcp",
                          "styles" : "grid",
                          "crs" : "EPSG:4283",
                          "format" : "png",
                          "time" : "Default",
                          "time_index" : "Default",
                        "source_url" : "http://localhost:8001/atmos_latest.nc",
                          "color_scale_range" : ["auto", ],
                          "n_colors" : ["7", ],
                          "palette" : "jet"
                        }
        
    def test_get_contour(self):
        self.param["styles"] = ["contour", ]
        self.assertIsNotNone(pc.get_contour(self.param, self.defaults))
        
    def test_get_legend(self):
        self.assertIsNotNone(pc.get_legend(self.param, self.defaults))
        
    def test_get_full_figure(self):
        self.param["styles"] = ["grid_treshold", ]
        self.assertIsNotNone(pc.get_full_figure(self.param, self.defaults))
        
    def test_invalid_data_source(self):
        self.param["source_url"] = "http://localhost/test.lala"
        self.assertRaises(ex.DatasourceNotSupportedError, \
                          pc.PlottingController, self.param, self.defaults )


class TestParameterValidator(unittest.TestCase):
    
    def setUp(self):
        self.param = { "bbox" : {  "min_lat" : "-90.0",
                                   "min_lon" : "0.0",
                                   "max_lat" : "90.0",
                                   "max_lon" : "360.0" 
                                } ,
                        "layers" : ["hr24_prcpa", ] ,
                        "styles" : ["grid", ] ,
                        "crs" : {   "name" : "EPSG" ,
                                    "identifier" : "4283" 
                                } ,
                        "width" : "300" ,
                        "height" : "400" ,
                        "format" : "png" ,
                        
                        "time" : "Default" ,
                        "time_index" : "Default" ,
                        "source_url" : "http://localhost:8001/atmos_latest.nc",
                        "color_scale_range" : ["-10", "10", ] ,
                        "n_colors" : ["10", ] ,
                        "palette" : "jet"
                     }
        self.defaults = { "request":"GetMap",
                          "version":"0.0.1",
                          "bbox" : {  "min_lat" : "-90.0",
                                      "min_lon" : "-180.0",
                                      "max_lat" : "90.0",
                                      "max_lon" : "180.0" 
                                } ,
                          "width" : "256",
                          "height" : "256",
                          "layers" : "hr24_prcp",
                          "styles" : "grid",
                          "crs" : "EPSG:4283",
                          "format" : "png",
                          "time" : "Default",
                          "time_index" : "Default",
                        "source_url" : "http://localhost:8001/atmos_latest.nc",
                          "color_scale_range" : ["auto", ],
                          "n_colors" : ["7", ],
                          "palette" : "jet"
                        }
    
    # 1) Check when everything is valid
    def test_all_valid(self):
        v = pc.ParameterValidator(self.param, self.defaults)
        v.validate()
    
    
    # 2) Test for absence of mandatory parameters
    def test_missing_bbox(self):
        del self.param["bbox"]
        v = pc.ParameterValidator(self.param, self.defaults)
        v.validate()
        self.assertEquals(self.param["bbox"], \
                          self.defaults["bbox"])
        
        
    def test_missing_layers(self):
        del self.param["layers"]
        v = pc.ParameterValidator(self.param, self.defaults)
        v.validate()
        self.assertEquals(self.param["layers"], \
                          self.defaults["layers"])
        
        
    def test_missing_styles(self):
        del self.param["styles"]
        v = pc.ParameterValidator(self.param, self.defaults)
        v.validate()
        self.assertEquals(self.param["styles"], \
                          self.defaults["styles"])
        
        
    def test_missing_crs(self):
        del self.param["crs"]
        v = pc.ParameterValidator(self.param, self.defaults)
        v.validate()
        self.assertEquals(self.param["crs"], \
                          self.defaults["crs"])
        
        
    def test_missing_height(self):
        del self.param["height"]
        v = pc.ParameterValidator(self.param, self.defaults)
        v.validate()
        self.assertEquals(self.param["height"], \
                          (int)(self.defaults["height"]))
        
        
    def test_missing_width(self):
        del self.param["width"]
        v = pc.ParameterValidator(self.param, self.defaults)
        v.validate()
        self.assertEquals(self.param["width"], \
                          (int)(self.defaults["width"]))
        
        
    def test_missing_format(self):
        del self.param["format"]
        v = pc.ParameterValidator(self.param, self.defaults)
        v.validate()
        self.assertEquals(self.param["format"], \
                          self.defaults["format"])
        
        
    def test_missing_source_url(self):
        del self.param["source_url"]
        v = pc.ParameterValidator(self.param, self.defaults)
        v.validate()
        self.assertEquals(self.param["source_url"], \
                          self.defaults["source_url"])
        
        
    def test_missing_color_scale_range(self):
        del self.param["color_scale_range"]
        v = pc.ParameterValidator(self.param, self.defaults)
        v.validate()
        self.assertEquals(self.param["color_scale_range"], \
                          [-4,4])
        
        
    def test_missing_number_of_colors(self):
        del self.param["n_colors"]
        v = pc.ParameterValidator(self.param,self.defaults)
        v.validate()
        self.assertEquals(self.param["n_colors"], \
                          list( int(i) for i in (self.defaults["n_colors"])))
        
        
    # 3) Check if correct value is assigned if optional parameter is missing
    def test_missing_time(self):
        del self.param["time"]
        v = pc.ParameterValidator(self.param, self.defaults)
        v.validate()
        self.assertEquals(self.param["time"], \
                          self.defaults["time"])
        
    def test_missing_time_index(self):
        del self.param["time_index"]
        v = pc.ParameterValidator(self.param, self.defaults)
        v.validate()
        self.assertEquals(self.param["time_index"], \
                          self.defaults["time_index"])
        
    def test_missing_palette(self):
        del self.param["palette"]
        v = pc.ParameterValidator(self.param, self.defaults)
        v.validate()
        self.assertEquals(self.param["palette"], \
                          self.defaults["palette"])
        
        
    # 4) Check for invalid numerical values
    def test_invalid_width(self):
        self.param["width"] = "a1024"
        v = pc.ParameterValidator(self.param, self.defaults)
        self.assertRaises(ex.InvalidParameterValueError, v.validate)
        
    def test_invalid_height(self):
        self.param["height"] = "e68"
        v = pc.ParameterValidator(self.param, self.defaults)
        self.assertRaises(ex.InvalidParameterValueError, v.validate)
        
    def test_invalid_color_scale_range(self):
        self.param["color_scale_range"] = [ "-10a", "-34a"]
        v = pc.ParameterValidator(self.param, self.defaults)
        self.assertRaises(ex.InvalidParameterValueError, v.validate)
        
    def test_invalid_n_colors(self):
        self.param["n_colors"] = [ "10a",]
        v = pc.ParameterValidator(self.param, self.defaults)
        self.assertRaises(ex.InvalidParameterValueError, v.validate)
        
    # 5) Validate Custom colors
    #TODO: Write tests