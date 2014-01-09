void draw() {
background(0);

 translate(width/2,height/2,-300);
rotateX(radians(45));
//rotateZ(radians(60));
rotateZ(radians(mouseY));
//rotateZ(radians(60));
//rotateY(radians(mouseX));

stroke(255);
strokeWeight(5);
line(0,0,-800,0,0,450);
  for (int i = 0; i<lilen; i++){
  li[i].display();

}


}


/*void controlEvent(ControlEvent theEvent) {
  switch(theEvent.controller().id()) {
    case(1):
     count = (int)(theEvent.controller().value());
    break;
    case(2):
     speed = (int)(theEvent.controller().value());
    break;

  }
  
}*/
