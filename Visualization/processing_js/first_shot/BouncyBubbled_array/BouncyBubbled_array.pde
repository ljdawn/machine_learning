#Bouncy Bubbles  
#based on code from Keith Peters. 
#Multiple-object collision.
#coffee script ver
#2014.2.23 xuanzhang
setup: ->
  size 600, 600
  fill 0
  noStroke()
  
  @brr = []
  for i in [0...3]
    @brr.push new Ball(width/2, (height/2)+i*50, 50+i*20)
draw: ->
  background 0
  for ball in [0...3]
    @brr[ball].collide(0, 0)
    @brr[ball].move()
    @brr[ball].display()
    
