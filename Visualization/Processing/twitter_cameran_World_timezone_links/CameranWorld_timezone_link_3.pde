void draw() {
  image(MapImage, 0, 0, width, height);
  stroke(0, 40);
  if (count < 100){
  for (int i = 0; i<count; i++){
  li[i].display(count-i);
 
  }

  }
  else if(count >= 100 && count < lilen){
    for (int i = count-100; i<count; i++){
  li[i].display(count-i); 
    }
  
  }else{
    for (int i = count-100; i<lilen; i++){
  //li[i].update();    
  li[i].display(count-i);
  }
  }
 if (count<lilen){
  fill(255,255,225,100);
   textSize(32);
  text(li[count].getd(),55,567);
  textSize(40);
  if(count < 100){
  text(count, 475, 570);}
  else if(count < 1000){text(count, 465, 570);}
  else if(count < 10000){text(count, 455, 570);}
  else if(count < 100000){text(count, 435, 570);}
  else {text(count, 425, 570);}
  count += speed;
  if(speed < 10){text(speed,690,570);
   }else if(speed < 100){text(speed,675,570);
   }else if(speed < 1000){text(speed,660,570);}


  textSize(14);
  fill(255,255,225,100);
  text("Cameran via Twitter", 330, 20);
  text("TIME", 185, 515);
  text("TWEETS", 475, 515);
  text("SPEED", 680, 515);
  fill(255,255,255,30);
  noStroke();
  rect(0, 0, 800, 30);
  noStroke();
  rect(0, 500, 800, 800);
  stroke(0);
  line(0,520,800,520);
  line(400, 500, 400, 600);
  line(600, 500, 600, 600);
  fill(255);
} }

void controlEvent(ControlEvent theEvent) {
  switch(theEvent.controller().id()) {
    case(1):
     count = (int)(theEvent.controller().value());
    break;
    case(2):
     speed = (int)(theEvent.controller().value());
    break;

  }
  
}


