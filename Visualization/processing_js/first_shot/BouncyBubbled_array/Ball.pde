class Ball
    
    constructor: ( @xpos, @ypos, @diamter ) ->
        @fri = -0.5
        @fur = 0.9
        @g = 0.1
        @vx = 20
        @vy = 1
    collide: (@ax, @ay) ->
        @vx += @ax
        @vy += @ay
    move: ->
        if @ypos == height - @diamter/2
            @vx *= @fur
        if abs(@vx) < 0.001
            @vx = 0
        @vy += @g
        @xpos += @vx
        @ypos += @vy
        if (@xpos + @diamter/2) > width
            @xpos = width - @diamter/2
            @vx *= @fri
        else if (@xpos - @diamter/2) < 0
            @xpos = @diamter/2
            @vx *= @fri
        if (@ypos + @diamter/2) > height
            @ypos = height - @diamter/2
            @vy *= @fri
        else if (@ypos - @diamter/2) < 0
            @ypos = @diamter/2
            @vy *= @fri                
        #println @vx
    display: ->
        fill 255, 50
        ellipse(@xpos, @ypos, @diamter, @diamter)
    
        
