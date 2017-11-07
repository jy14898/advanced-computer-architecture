#!/usr/bin/env python2
# ~/firefox-lab/firefox -p "Firefox Lab" -no-remote "http://localhost:8080/"
import sys

sys.dont_write_bytecode = True

import os
import cherrypy

from programs.add_test import program
from processor import Processor

import threading

path   = os.path.abspath(os.path.dirname(__file__))
config = {
    'global' : {
        'server.socket_host' : '127.0.0.1',
        'server.socket_port' : 8080,
        'server.thread_pool' : 8
    },
    '/': {
        'tools.staticdir.on': True,
        'tools.staticdir.dir': '',
        'tools.staticdir.root': os.path.join(path,'public'),
        'tools.staticdir.index': 'index.html'
    },
    '/processor': {
        'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
        'tools.response_headers.on': True,
        'tools.response_headers.headers': [('Content-Type', 'application/json')],
    }
}

processor = Processor(program["instructions"], program["data"])
processor_lock = threading.Lock()

@cherrypy.expose
class ProcessorStructureAPI(object):
    @cherrypy.tools.json_out()
    def GET(self):
        # don't need lock as this never changes after construction
        structure = {
            "components": {},
            "connections": []
        }

        for name, component in processor.object_lookup.iteritems():
            structure["components"][name] = {
                "type": type(component).__name__,
            }

            structure["components"][name].update(component.config)

        for (out_component, out_key),(in_component, in_key) in processor.connections:
            structure["connections"].append({
                "out_component": processor.get_name(out_component),
                "out_key": out_key,
                "in_component": processor.get_name(in_component),
                "in_key": in_key
            })

        return structure

@cherrypy.expose
class ProcessorAPI(object):
    structure = ProcessorStructureAPI()

    @staticmethod
    def collect_state():
        state = {
            "components": {}
        }

        with processor_lock:
            for name, component in processor.object_lookup.iteritems():
                state["components"][name] = {}

                state["components"][name]["input_values"]  = dict( (d, processor.output_value_cache[(a,b)]) for ((a,b),(c,d)) in processor.connections if c == component)
                state["components"][name]["output_values"] = dict((out_key,processor.output_value_cache[(component, out_key)]) for out_key in component.config["output_keys"])

                state["components"][name].update(component.state)

        state["phase"] = processor.clock_phase

        return state

    @cherrypy.tools.json_out()
    def GET(self):
        return ProcessorAPI.collect_state()

    @cherrypy.tools.json_out()
    def POST(self, steps=0):
        for _ in xrange(int(steps)):
            with processor_lock:
                processor.step()
        return ProcessorAPI.collect_state()

class Root(object):
    processor = ProcessorAPI()


if __name__ == '__main__':
  cherrypy.tree.mount(Root(), '/', config)
  cherrypy.engine.start()
  cherrypy.engine.block()
