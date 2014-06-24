class MRect
    constructor: ( @w, @xpos, @h, @ypos, @d, @t ) ->
    move: ( posX, posY, damping ) ->
        dif = @ypos - posY
        @ypos -= dif/damping if abs(dif) > 1
        dif = @xpos - posX
        @xpos -= dif/damping if abs(dif) > 1
    display: ->
        for i in [0...@t]
            rect @xpos+(i*(@d+@w)), @ypos, @w, height*@h
