#All Examples Written by Casey Reas and Ben Fry
setup: ->
    size 200, 200
    stroke 255 
    @y = 100
    noLoop()
draw: ->
    background 0
    @y -= 1
    if @y < 0
        @y = height
    line 0, @y, width, @y

mousePressed: ->
    redraw()

