class Line{
float x1, y1, x2, y2;
String da, stl, enl;
int count;
Line(float xp1, float yp1, float xp2, float yp2, String dat, String st, String en) {
x1 = xp1; 
y1 = yp1; 
x2 = xp2;
y2 = yp2;
da = dat;
stl = st;
enl = en;
}

void display(int count){

//fill(255,0,255,100);
if(x1 != x2 && y1 != y2){
stroke(255,255,255,100);
strokeWeight(10);
//strokeCap(ROUND);
line(x1+(count-1)*((x2-x1)/100),y1+(count-1)*((y2-y1)/100),x1+count*((x2-x1)/100),y1+count*((y2-y1)/100));
stroke(255,255,255,100);
strokeWeight(1);
//line(x1,y1,x2,y2);
fill(255,255,255,100);
smooth();
noStroke();
ellipse(x1, y1, 5, 5);
//ellipse(x2, y2, 2, 2);


for (int j = 0; j<10; j++){
   smooth();
   fill(255,255,25,10-j);
   ellipse(x1, y1, 2*j, 2*j);

   ellipse(x2, y2, 2*j, 2*j);
 }
if(count<=60){ 
//for (int d = 0; d < ix; d++){
   smooth();
   //fill(255,255,25,50-ix);
   stroke(25,255,25,60-count);

   strokeWeight(3);
   ellipse(x1, y1, count*3, count*3);
   stroke(255,255,255,30-count);
   ellipse(x2, y2, count, count);
   fill(255,255,255,100-count);
   textSize(10);
   text(stl, x1,y1);
   text(enl,x2,y2);

 //}
 }
 
}else{
fill(255,255,255,100);
smooth();
noStroke();
ellipse(x1, y1, 5, 5);
for (int j = 0; j<10; j++){
   smooth();
   fill(255,255,255,10-j);
   ellipse(x1, y1, 2*j, 2*j);
 }


} 


}


String getd(){return da;}
}

