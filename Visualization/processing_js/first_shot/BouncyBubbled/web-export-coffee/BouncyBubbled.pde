#Bouncy Bubbles  
#based on code from Keith Peters. 
#Multiple-object collision.
#coffee script ver0.1
#2014.2.24 xuanzhang
#with x->friction
setup: ->
  size 600, 600
  fill 0
  noStroke()
  @ball_num = 12
  @spring = 0.5
  @brr = []
  for i in [0...@ball_num]
    @brr.push new Ball(width/2-i*100, (height/2)+i*50, i*2+50)
draw: ->
  background 0
  for ball in [0...@ball_num]
    for ith in [ball+1...@ball_num]
      @dx = @brr[ith].xpos - @brr[ball].xpos
      @dy = @brr[ith].ypos - @brr[ball].ypos
      @dist = sqrt(@dx*@dx + @dy*@dy)
      @mind = @brr[ith].diamter/2 + @brr[ball].diamter/2
      if @dist < @mind
        @angle = atan2(@dy, @dx)
        @tarX = @brr[ball].xpos + cos(@angle) * @mind
        @tarY = @brr[ball].ypos + sin(@angle) * @mind
        @tax = (@tarX - @brr[ith].xpos) * @spring
        @tay = (@tarY - @brr[ith].ypos) * @spring
        @brr[ith].collide(@tax, @tay)
        @brr[ball].collide(-@tax, -@tay)
    @brr[ball].move()
    @brr[ball].display()
    
