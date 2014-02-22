#All Examples Written by Casey Reas and Ben Fry
setup: ->
    size 200, 200
    background 51
    noStroke()
    smooth()
    noLoop()
draw: ->
    draw_target 68, 34, 200, 10
    draw_target 152, 16, 100, 3
    draw_target 100, 144, 80, 5

draw_target = (xloc, yloc, size, num) ->
    grayvalues = 255/num
    steps = size/num
    for i in [0...num]
        fill i, grayvalues
        ellipse xloc, yloc, size-i*steps, size-i*steps
