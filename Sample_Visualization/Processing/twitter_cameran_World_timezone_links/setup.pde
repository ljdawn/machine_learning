void setup() {
  size(800, 600);
  smooth();
  
  MapImage = loadImage("worldmap.png");
  mercatorMap = new MercatorMap(800, 600,85.0511,-85.0511,-180,180);
  controlP5 = new ControlP5(this);
  controlP5.addSlider("TWEETS [1/140000]",0,140000,count,10,480,690,10).setId(1);
  controlP5.addSlider("SPEED [0/200]",0,200,speed,10,460,690,10).setId(2);


  
li = new Line[lilen];
String[] lis = loadStrings("lingage_geo_for_processing.csv");
 for(int i=0; i<lilen; i++){
String[] lia= lis[i].split(",");
 float r1 = Float.parseFloat(lia[0]);
 float r2 = Float.parseFloat(lia[1]);
 float r3 = Float.parseFloat(lia[2]);
 float r4 = Float.parseFloat(lia[3]);
 String x = lia[4];
 String s = lia[5];
 String e = lia[6];
 tokyo1 = mercatorMap.getScreenLocation(new PVector(r1, r2));
 tokyo2 = mercatorMap.getScreenLocation(new PVector(r3, r4));
 li[i] = new Line(tokyo1.x, tokyo1.y,tokyo2.x, tokyo2.y,x,s,e);
 }

frameRate(60);
}
