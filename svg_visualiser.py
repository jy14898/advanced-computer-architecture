#!/usr/bin/env python2

from processor import Processor
from collections import namedtuple

'''
Create GUIComponent for all types
All things need to be multiple of 10s for grid snapping when in inkscape (right bar remember)
'''


def Component(component, width):
    height = (max(len(component.output_keys),len(component.input_keys))+1)*10

    svg = ""

    svg += '<rect class="main" x="0" y="0" width="{}" height="{}"/>'.format(width, height)

    for i, v in enumerate(component.input_keys):
        svg += '<text class="input_label" data-key="{}" x="2" y="{}" >{}</text>'.format(v, (i+1)*10 + 1.5, v)
        svg += \
        '''
        <line x1="-5" y1="{0}" x2="0" y2="{0}" stroke-width="1" stroke="#999999"/>
        '''.format((i+1)*10)

    for i, v in enumerate(component.output_keys):
        svg += '<text class="output_label" data-key="{}" x="{}" y="{}"  text-anchor="end">{}</text>'.format(v, width - 2, (i+1)*10 + 1.5, v)
        svg += \
        '''
        <line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke-width="1" stroke="#999999"/>
        '''.format(width, (i+1)*10, width+5)

    if component.clocked:
        d = "M {} {} m-5 0 l5 -5 l5 5".format(width/2, height)
        svg += '<path d="{}" fill="none" stroke-width="1" stroke="#999999"/>'.format(d)

    return svg #(svg,)

def PipelineBuffer(component, width):
    height = (max(len(component.output_keys),len(component.input_keys))+1)*10

    svg = ""

    svg += '<rect class="main" x="0" y="0" width="{}" height="{}"/>'.format(width, height)

    for i, v in enumerate(["reset","freeze"]):
        svg += '<text class="input_label" data-key="{}" x="2" y="{}" >{}</text>'.format(v, (i+1)*10 + 1.5, v)
        svg += \
        '''
        <line x1="-5" y1="{0}" x2="0" y2="{0}" stroke-width="1" stroke="#999999"/>
        '''.format((i+1)*10)

    svg += '<line x1="0" y1="25" x2="{}" y2="25" stroke-width="1" stroke="#999999"/>'.format(width)


    for i, v in enumerate(component.names.keys()):
        svg += '<text class="input_label output_label" data-key="{}" x="{}" y="{}" text-anchor="middle" >{}</text>'.format(v, width/2, (i+3)*10 + 1.5, v)
        svg += \
'''
<line x1="-5" y1="{1}" x2="0" y2="{1}" stroke-width="1" stroke="#999999"/>
<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke-width="1" stroke="#999999"/>
'''.format(width,(i+3)*10, width+5)

    if component.clocked:
        d = "M {} {} m-5 0 l5 -5 l5 5".format(width/2, height)
        svg += '<path d="{}" fill="none" stroke-width="1" stroke="#999999"/>'.format(d)

    return svg #(svg,)

template_map = {
    "Latch": Component,
    "Component": Component,
    "Incrementer": Component,
    "Multiplexer": Component,
    "InstructionMemory": Component,
    "PipelineBuffer": PipelineBuffer,
    "InstructionSplit": Component,
    "Decoder": Component,
    "RegisterFile": Component,
    "Constant": Component,
    "ALU": Component,
    "And": Component,
    "DataMemory": Component,
    "Or": Component,
}

prop = namedtuple("property", "width")

property_map = {
    "default": prop(width=90),
    "inc_pc": prop(width=30),
    "pc": prop(width=50),
    "pc_mux": prop(width=50),
    "decode_buffer": prop(width=50),
    "instruction_split": prop(width=80),
    "decoder": prop(width=80),
    "register_file": prop(width=100),
    "execute_buffer_reset": prop(width=20),
    "execute_buffer": prop(width=70),
    "alu": prop(width=70),
    "alu_source_mux": prop(width=60),
    "branch_addr_mux": prop(width=60),
    "wb_value_mux": prop(width=60),
    "freeze_pipeline": prop(width=60),
    "do_branch": prop(width=60),
    "mem_buffer_reset": prop(width=20),
    "mem_buffer": prop(width=70),
    "data_memory": prop(width=80),
    "wb_buffer_reset": prop(width=20),
    "wb_buffer": prop(width=70),
}

def visualise(proc):
    style = \
'''
<style>

.component .main {
    fill: #FFFFFF;
    stroke: #999999;
}

.connection {
    fill: none;
    stroke-width: 1;
    stroke: #999999;
}

.connection:hover {
    stroke-width: 2;
    stroke: #FF0000;
}

.input_label:hover, .output_label:hover {
    fill: #FF0000;
}


</style>
'''
    svg = '<svg id="processor" xmlns="http://www.w3.org/2000/svg" font-family="monospace" font-size="6">'
    svg += style
    svg += '<g class="svg-pan-zoom_viewport">'
    x = 0
    for component in proc.components:
        x += 120
        svg += "<g transform=\"translate({} 0)\" class=\"component\" data-id=\"{}\" data-clocked=\"{}\">".format(x,proc.get_name(component), "true" if component.clocked else "false")
        svg += \
        '''
        <text x="0" y="-14">
            {}
        </text>
        <text x="0" y="-4">
            ({})
        </text>
        '''.format(proc.get_name(component), type(component).__name__)

        if proc.get_name(component) in property_map:
            properties = property_map[proc.get_name(component)]
        else:
            properties = property_map["default"]


        svg += template_map[type(component).__name__](component, *properties)
        svg += "</g>"

    # x = 0
    # for (out_component, out_key),(in_component, in_key) in proc.connections:
    #     d = "M0 {} l20 0".format(x)
    #     svg += '<path d="{}" class="connection" data-from="[{},{}]" data-to="[{},{}]" />'.format(d, proc.get_name(out_component), out_key, proc.get_name(in_component), in_key)
    #     svg += '<text x="-5" y="{}" text-anchor="end">{}[{}]</text>'.format(x, proc.get_name(out_component), out_key)
    #     svg += '<text x="25" y="{}">{}[{}]</text>'.format(x, proc.get_name(in_component), in_key)
    #     x += 20

    svg += "</g>"
    svg += "</svg>"

    with open("out.svg","w") as svgfile:
        svgfile.write(svg)


visualise(Processor([],[]))
