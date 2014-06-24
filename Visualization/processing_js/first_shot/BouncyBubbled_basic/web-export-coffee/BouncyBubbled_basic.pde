#Bouncy Bubbles  
#based on code from Keith Peters. 
#Multiple-object collision.
#coffee script ver
#2014.2.23 xuanzhang
setup: ->
  size 600, 600
  fill 0
  noStroke()
  arr = [0, 1, 2]
  @brr = []
  for i in arr
    @brr.push new Ball(width/2, (height/2)+i*50, 50+i*20)
draw: ->
  background 0
  for ball in @brr
    ball.collide()
    ball.move()
    ball.display()
