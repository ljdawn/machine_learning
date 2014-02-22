#All Examples Written by Casey Reas and Ben Fry
setup: ->
    size 200, 200
    background 127
    noStroke
draw: ->
    for i in [0...height] by 20
        # step is set by 'by'
        fill 0
        rect 0, i, width, 10
        fill 255
        rect i, 0, 10, height


