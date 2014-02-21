class Spot {
float x, y, diameter, i;
String da;
Spot(float xpos, float ypos, float dia, String dat) {
x = xpos; 
y = ypos; 
diameter = dia;
da = dat;

}

float getx(){return x;}
float gety(){return y;}
String getd(){return da;}
float geto(){return diameter;}
float geti(){return i;}

void display(float ix) {
fill(255,255,255,100);
smooth();
noStroke();
ellipse(x, y, diameter, diameter);

for (int j = 0; j<10; j++){
   smooth();
   fill(255,255,25,10-j);
   ellipse(x, y, diameter*j, diameter*j);

 }

if(ix<=100){ 
//for (int d = 0; d < ix; d++){
   smooth();
   //fill(255,255,25,50-ix);
   stroke(25,255,25,100-ix);
   ellipse(x, y, diameter*ix*0.5, diameter*ix*0.5);

 //}
 }
 
}

}
  
