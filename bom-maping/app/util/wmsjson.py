import sys
import re
import json

# TODO : JSON Output using Templates
# template = xml.templateLoader.load('wms_capabilities_json.txt',
                                                       #cls=NewTextTemplate)
def output(contents):
    json = JsonGenerator()
    return json.output(contents)

class JsonGenerator(object):
    """ JSON Utiltiy 
    
    This class is responsible for all json related functionality.
    """

    def __init__(self):
        """ Constructor """
        pass

    def make_json(self,dict):
        return json.JSONDecoder(dict)

    def xml2dict(xml, jsonString = ''):
        tags, keys = re.findall('</?[A-Za-z0-9]+>',xml), []
        for tag in tags: keys.append(re.sub('[</>]+','',tag))
        for index in range(len(tags)-1):
            jsonString += {'<><>':   '"'+keys[index]+'": {',
                       '<></>':  '"'+keys[index]+'": "'+xml[xml.find(tags[index])+len(tags[index]):xml.find(tags[index+1])]+'"',
                       '</><>':  ', ',
                       '</></>': '}'}[tags[index].replace(keys[index],'')+tags[index+1].replace(keys[index+1],'')]
        return json.loads('{%s}' % jsonString)

    def output(self,contents):
        #return self.make_json(dict)
        return json.dumps(contents)
