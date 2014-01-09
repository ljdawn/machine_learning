
void setup() {
  size(width,height, P3D);
  smooth();
 // controlP5 = new ControlP5(this);
  //controlP5.addSlider("EDGES [1/9900]",0,9900,count,10,680,690,10).setId(1);
  //controlP5.addSlider("SPEED [0/200]",0,200,speed,10,660,690,10).setId(2);
  
  
li = new Line[lilen];
String[] lis = loadStrings("forpro_buzz_3.csv");
 for(int i=0; i<lilen; i++){
String[] lia= lis[i].split(",");
 float r1 = Float.parseFloat(lia[0]);
 float r2 = Float.parseFloat(lia[1]);
 float r3 = Float.parseFloat(lia[2]);
 float r4 = Float.parseFloat(lia[3]);
 float r5 = Float.parseFloat(lia[4]);
 float r6 = Float.parseFloat(lia[5]);
 String n1 = lia[6];
 String n2 = lia[7];
 String n3 = lia[8];
  float r = Float.parseFloat(lia[9]);
 float g = Float.parseFloat(lia[10]);
 float b = Float.parseFloat(lia[11]);
  float s = Float.parseFloat(lia[12]);
 li[i] = new Line(r1-595.13, r2-494.56, r3, r4-595.13, r5-494.56,r6,n1,n2,n3,r,g,b,s);
 }

frameRate(60);
}
