class Line{
float x1, y1, z1, x2, y2, z2, r, g, b, s;
String n1, n2, t;
Line(float xp1, float yp1, float zp1, float xp2, float yp2, float zp2,String na1, String na2, String ti, float r1,float g1,float b1, float s1) {
x1 = xp1; 
y1 = yp1; 
z1 = zp1; 
x2 = xp2;
y2 = yp2;
z2 = zp2;
n1 = na1;
n2 = na2;
t = ti;
r = r1;
b = b1;
g = g1;
s = s1;
}

void display(){
stroke(r,g,b,70);
strokeWeight(1);

line(x1,y1,z1,x2,y2,z2);
pushMatrix();
translate(x1,y1,z1); 
rotateY(0.5);
fill(r,g,b);
box(s);
popMatrix();
/*pushMatrix();
translate(x2,y2,z2); 
rotateY(0.5);
fill(r,g,b);
box(s);
popMatrix();*/

}









String gettime(){
return t;}

}
