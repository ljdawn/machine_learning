###
###

setup: ->
  size 600, 600, P3D

draw: ->

  background 0
  
  translate mouseX, mouseY, (mouseX+mouseY)/2
  rotateX mouseX/100
  box 90
  stroke 255
  fill 0
