#Bouncy Bubbles  
#based on code from Keith Peters. 
#Multiple-object collision.
#coffee script ver
#2014.2.23 xuanzhang
setup: ->
  size 600, 600
  fill 0
  noStroke()
  @spring = 0.05
  @brr = []
  for i in [0...2]
    @brr.push new Ball(width/2-i*100, (height/2)+i*50, 50)
draw: ->
  background 0
  for ball in [0...2]
    for ith in [ball+1...2]
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
    
