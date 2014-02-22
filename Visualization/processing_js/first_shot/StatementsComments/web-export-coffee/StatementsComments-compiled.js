var SketchStatementscomments;

SketchStatementscomments = (function() {
    var ADD=ALIGN_CENTER=ALIGN_LEFT=ALIGN_RIGHT=ALPHA=ALPHA_MASK=ALT=AMBIENT=ARGB=ARROW=BACKSPACE=BASELINE=BEVEL=BLEND=BLUE_MASK=BLUR=BOTTOM=BURN=CENTER=CHATTER=CLOSE=CMYK=CODED=COMPLAINT=COMPONENT=COMPOSITE=CONCAVE_POLYGON=CONTROL=CONVEX_POLYGON=CORNER=CORNERS=CROSS=CUSTOM=DARKEST=DEGREES=DEG_TO_RAD=DELETE=DIAMETER=DIFFERENCE=DIFFUSE=DILATE=DIRECTIONAL=DISABLED=DODGE=DOWN=DXF=ENTER=EPSILON=ERODE=ESC=EXCLUSION=GIF=GRAY=GREEN_MASK=GROUP=HALF=HALF_PI=HAND=HARD_LIGHT=HINT_COUNT=HSB=IMAGE=INVERT=JAVA2D=JPEG=
    LEFT=LIGHTEST=LINES=LINUX=MACOSX=MAX_FLOAT=MAX_INT=MITER=MODEL=MOVE=MULTIPLY=NORMAL=NORMALIZED=NO_DEPTH_TEST=NTSC=ONE=OPAQUE=OPEN=OPENGL=ORTHOGRAPHIC=OVERLAY=P2D=P3D=PAL=PDF=PERSPECTIVE=PI=PIXEL_CENTER=POINT=POINTS=POSTERIZE=PROBLEM=PROJECT=QUADS=QUAD_STRIP=QUARTER_PI=RADIANS=RADIUS=RAD_TO_DEG=RED_MASK=REPLACE=RETURN=RGB=RIGHT=ROUND=SCREEN=SECAM=SHIFT=SOFT_LIGHT=SPECULAR=SQUARE=SUBTRACT=SVIDEO=TAB=TARGA=TEXT=TFF=THIRD_PI=THRESHOLD=TIFF=TOP=TRIANGLES=TRIANGLE_FAN=TRIANGLE_STRIP=TUNER=TWO=TWO_PI=UP=
    WAIT=WHITESPACE=XML=ArrayList=BufferedReader=Character=HashMap=Integer=PFont=PGraphics=PImage=PShader=PShape=PVector=PrintWriter=StringBuffer=abs=acos=addChild=alpha=ambient=ambientLight=append=applyMatrix=arc=asin=atan=atan2=background=beginCamera=beginContour=beginRaw=beginRecord=beginShape=bezier=bezierDetail=bezierPoint=bezierTangent=bezierVertex=binary=bind=blend=blendColor=blendMode=blue=box=breakShape=brightness=cache=camera=ceil=clip=color=colorMode=concat=constrain=copy=cos=createFont=createGraphics=
    createImage=createInput=createOutput=createPath=createReader=createShape=createWriter=cursor=curve=curveDetail=curvePoint=curveTangent=curveTightness=curveVertex=day=degrees=directionalLight=dist=draw=ellipse=ellipseMode=emissive=end=endCamera=endContour=endRaw=endRecord=endShape=exit=exp=expand=fill=filter=floor=focused=frustum=get=green=hex=hint=hour=hue=image=imageMode=join=keyPressed=keyReleased=keyTyped=lerp=lerpColor=lightFalloff=lightSpecular=lights=line=loadBytes=loadFont=loadImage=loadMatrix=
    loadPixels=loadShader=loadShape=loadStrings=loadType=log=doLoop=mag=map=match=matchAll=max=millis=min=minute=modelX=modelY=modelZ=month=mouseButton=mouseClicked=mouseDragged=mouseMoved=mousePressed=mouseReleased=nf=nfc=nfp=nfs=noClip=noCursor=noFill=noHint=noLights=noLoop=noSmooth=noStroke=noTint=noise=noiseDetail=noiseSeed=norm=normal=open=openStream=ortho=parseByte=perspective=point=pointLight=popMatrix=popStyle=pow=print=printCamera=printMatrix=printProjection=println=pushMatrix=pushStyle=quad=
    quadraticVertex=radians=random=randomSeed=rect=rectMode=red=redraw=requestImage=resetMatrix=resetShader=reverse=rotate=rotateX=rotateY=rotateZ=round=saturation=save=saveBytes=saveFile=saveFrame=savePath=saveStream=saveStrings=saveType=scale=screenX=screenY=screenZ=second=selectFolder=selectInput=selectOutput=set=setup=shader=shape=shapeMode=shearX=shearY=shininess=shorten=sin=sketchFile=sketchPath=smooth=sort=specular=sphere=sphereDetail=splice=split=splitTokens=spotLight=sq=sqrt=start=stop=stroke=
    strokeCap=strokeJoin=strokeWeight=subset=tan=text=textAlign=textAscent=textDescent=textFont=textLeading=textMode=textSize=textWidth=texture=textureMode=tint=translate=triangle=trim=unbinary=unhex=updatePixels=vertex=year=null,injectProcessingApi=function(a){ADD=a.ADD;ALIGN_CENTER=a.ALIGN_CENTER;ALIGN_LEFT=a.ALIGN_LEFT;ALIGN_RIGHT=a.ALIGN_RIGHT;ALPHA=a.ALPHA;ALPHA_MASK=a.ALPHA_MASK;ALT=a.ALT;AMBIENT=a.AMBIENT;ARGB=a.ARGB;ARROW=a.ARROW;BACKSPACE=a.BACKSPACE;BASELINE=a.BASELINE;BEVEL=a.BEVEL;BLEND=a.BLEND;
    BLUE_MASK=a.BLUE_MASK;BLUR=a.BLUR;BOTTOM=a.BOTTOM;BURN=a.BURN;CENTER=a.CENTER;CHATTER=a.CHATTER;CLOSE=a.CLOSE;CMYK=a.CMYK;CODED=a.CODED;COMPLAINT=a.COMPLAINT;COMPONENT=a.COMPONENT;COMPOSITE=a.COMPOSITE;CONCAVE_POLYGON=a.CONCAVE_POLYGON;CONTROL=a.CONTROL;CONVEX_POLYGON=a.CONVEX_POLYGON;CORNER=a.CORNER;CORNERS=a.CORNERS;CROSS=a.CROSS;CUSTOM=a.CUSTOM;DARKEST=a.DARKEST;DEGREES=a.DEGREES;DEG_TO_RAD=a.DEG_TO_RAD;DELETE=a.DELETE;DIAMETER=a.DIAMETER;DIFFERENCE=a.DIFFERENCE;DIFFUSE=a.DIFFUSE;DILATE=a.DILATE;
    DIRECTIONAL=a.DIRECTIONAL;DISABLED=a.DISABLED;DODGE=a.DODGE;DOWN=a.DOWN;DXF=a.DXF;ENTER=a.ENTER;EPSILON=a.EPSILON;ERODE=a.ERODE;ESC=a.ESC;EXCLUSION=a.EXCLUSION;GIF=a.GIF;GRAY=a.GRAY;GREEN_MASK=a.GREEN_MASK;GROUP=a.GROUP;HALF=a.HALF;HALF_PI=a.HALF_PI;HAND=a.HAND;HARD_LIGHT=a.HARD_LIGHT;HINT_COUNT=a.HINT_COUNT;HSB=a.HSB;IMAGE=a.IMAGE;INVERT=a.INVERT;JAVA2D=a.JAVA2D;JPEG=a.JPEG;LEFT=a.LEFT;LIGHTEST=a.LIGHTEST;LINES=a.LINES;LINUX=a.LINUX;MACOSX=a.MACOSX;MAX_FLOAT=a.MAX_FLOAT;MAX_INT=a.MAX_INT;MITER=a.MITER;
    MODEL=a.MODEL;MOVE=a.MOVE;MULTIPLY=a.MULTIPLY;NORMAL=a.NORMAL;NORMALIZED=a.NORMALIZED;NO_DEPTH_TEST=a.NO_DEPTH_TEST;NTSC=a.NTSC;ONE=a.ONE;OPAQUE=a.OPAQUE;OPEN=a.OPEN;OPENGL=a.OPENGL;ORTHOGRAPHIC=a.ORTHOGRAPHIC;OVERLAY=a.OVERLAY;P2D=a.P2D;P3D=a.P3D;PAL=a.PAL;PDF=a.PDF;PERSPECTIVE=a.PERSPECTIVE;PI=a.PI;PIXEL_CENTER=a.PIXEL_CENTER;POINT=a.POINT;POINTS=a.POINTS;POSTERIZE=a.POSTERIZE;PROBLEM=a.PROBLEM;PROJECT=a.PROJECT;QUADS=a.QUADS;QUAD_STRIP=a.QUAD_STRIP;QUARTER_PI=a.QUARTER_PI;RADIANS=a.RADIANS;RADIUS=
    a.RADIUS;RAD_TO_DEG=a.RAD_TO_DEG;RED_MASK=a.RED_MASK;REPLACE=a.REPLACE;RETURN=a.RETURN;RGB=a.RGB;RIGHT=a.RIGHT;ROUND=a.ROUND;SCREEN=a.SCREEN;SECAM=a.SECAM;SHIFT=a.SHIFT;SOFT_LIGHT=a.SOFT_LIGHT;SPECULAR=a.SPECULAR;SQUARE=a.SQUARE;SUBTRACT=a.SUBTRACT;SVIDEO=a.SVIDEO;TAB=a.TAB;TARGA=a.TARGA;TEXT=a.TEXT;TFF=a.TFF;THIRD_PI=a.THIRD_PI;THRESHOLD=a.THRESHOLD;TIFF=a.TIFF;TOP=a.TOP;TRIANGLES=a.TRIANGLES;TRIANGLE_FAN=a.TRIANGLE_FAN;TRIANGLE_STRIP=a.TRIANGLE_STRIP;TUNER=a.TUNER;TWO=a.TWO;TWO_PI=a.TWO_PI;UP=a.UP;
    WAIT=a.WAIT;WHITESPACE=a.WHITESPACE;XML=a.XML;ArrayList=a.ArrayList;BufferedReader=a.BufferedReader;Character=a.Character;HashMap=a.HashMap;Integer=a.Integer;PFont=a.PFont;PGraphics=a.PGraphics;PImage=a.PImage;PShader=a.PShader;PShape=a.PShape;PVector=a.PVector;PrintWriter=a.PrintWriter;StringBuffer=a.StringBuffer;abs=a.abs;acos=a.acos;addChild=a.addChild;alpha=a.alpha;ambient=a.ambient;ambientLight=a.ambientLight;append=a.append;applyMatrix=a.applyMatrix;arc=a.arc;asin=a.asin;atan=a.atan;atan2=a.atan2;
    background=a.background;beginCamera=a.beginCamera;beginContour=a.beginContour;beginRaw=a.beginRaw;beginRecord=a.beginRecord;beginShape=a.beginShape;bezier=a.bezier;bezierDetail=a.bezierDetail;bezierPoint=a.bezierPoint;bezierTangent=a.bezierTangent;bezierVertex=a.bezierVertex;binary=a.binary;bind=a.bind;blend=a.blend;blendColor=a.blendColor;blendMode=a.blendMode;blue=a.blue;box=a.box;breakShape=a.breakShape;brightness=a.brightness;cache=a.cache;camera=a.camera;ceil=a.ceil;clip=a.clip;color=a.color;
    colorMode=a.colorMode;concat=a.concat;constrain=a.constrain;copy=a.copy;cos=a.cos;createFont=a.createFont;createGraphics=a.createGraphics;createImage=a.createImage;createInput=a.createInput;createOutput=a.createOutput;createPath=a.createPath;createReader=a.createReader;createShape=a.createShape;createWriter=a.createWriter;cursor=a.cursor;curve=a.curve;curveDetail=a.curveDetail;curvePoint=a.curvePoint;curveTangent=a.curveTangent;curveTightness=a.curveTightness;curveVertex=a.curveVertex;day=a.day;degrees=
    a.degrees;directionalLight=a.directionalLight;dist=a.dist;draw=a.draw;ellipse=a.ellipse;ellipseMode=a.ellipseMode;emissive=a.emissive;end=a.end;endCamera=a.endCamera;endContour=a.endContour;endRaw=a.endRaw;endRecord=a.endRecord;endShape=a.endShape;exit=a.exit;exp=a.exp;expand=a.expand;fill=a.fill;filter=a.filter;floor=a.floor;focused=a.focused;frustum=a.frustum;get=a.get;green=a.green;hex=a.hex;hint=a.hint;hour=a.hour;hue=a.hue;image=a.image;imageMode=a.imageMode;join=a.join;keyReleased=a.keyReleased;
    keyTyped=a.keyTyped;lerp=a.lerp;lerpColor=a.lerpColor;lightFalloff=a.lightFalloff;lightSpecular=a.lightSpecular;lights=a.lights;line=a.line;loadBytes=a.loadBytes;loadFont=a.loadFont;loadImage=a.loadImage;loadMatrix=a.loadMatrix;loadPixels=a.loadPixels;loadShader=a.loadShader;loadShape=a.loadShape;loadStrings=a.loadStrings;loadType=a.loadType;log=a.log;doLoop=a.loop;mag=a.mag;map=a.map;match=a.match;matchAll=a.matchAll;max=a.max;millis=a.millis;min=a.min;minute=a.minute;modelX=a.modelX;modelY=a.modelY;
    modelZ=a.modelZ;month=a.month;mouseButton=a.mouseButton;mouseClicked=a.mouseClicked;mouseDragged=a.mouseDragged;mouseMoved=a.mouseMoved;mouseReleased=a.mouseReleased;nf=a.nf;nfc=a.nfc;nfp=a.nfp;nfs=a.nfs;noClip=a.noClip;noCursor=a.noCursor;noFill=a.noFill;noHint=a.noHint;noLights=a.noLights;noLoop=a.noLoop;noSmooth=a.noSmooth;noStroke=a.noStroke;noTint=a.noTint;noise=a.noise;noiseDetail=a.noiseDetail;noiseSeed=a.noiseSeed;norm=a.norm;normal=a.normal;open=a.open;openStream=a.openStream;ortho=a.ortho;
    parseByte=a.parseByte;perspective=a.perspective;point=a.point;pointLight=a.pointLight;popMatrix=a.popMatrix;popStyle=a.popStyle;pow=a.pow;print=a.print;printCamera=a.printCamera;printMatrix=a.printMatrix;printProjection=a.printProjection;println=a.println;pushMatrix=a.pushMatrix;pushStyle=a.pushStyle;quad=a.quad;quadraticVertex=a.quadraticVertex;radians=a.radians;random=a.random;randomSeed=a.randomSeed;rect=a.rect;rectMode=a.rectMode;red=a.red;redraw=a.redraw;requestImage=a.requestImage;resetMatrix=
    a.resetMatrix;resetShader=a.resetShader;reverse=a.reverse;rotate=a.rotate;rotateX=a.rotateX;rotateY=a.rotateY;rotateZ=a.rotateZ;round=a.round;saturation=a.saturation;save=a.save;saveBytes=a.saveBytes;saveFile=a.saveFile;saveFrame=a.saveFrame;savePath=a.savePath;saveStream=a.saveStream;saveStrings=a.saveStrings;saveType=a.saveType;scale=a.scale;screenX=a.screenX;screenY=a.screenY;screenZ=a.screenZ;second=a.second;selectFolder=a.selectFolder;selectInput=a.selectInput;selectOutput=a.selectOutput;set=
    a.set;setup=a.setup;shader=a.shader;shape=a.shape;shapeMode=a.shapeMode;shearX=a.shearX;shearY=a.shearY;shininess=a.shininess;shorten=a.shorten;sin=a.sin;size=a.size;sketchFile=a.sketchFile;sketchPath=a.sketchPath;smooth=a.smooth;sort=a.sort;specular=a.specular;sphere=a.sphere;sphereDetail=a.sphereDetail;splice=a.splice;split=a.split;splitTokens=a.splitTokens;spotLight=a.spotLight;sq=a.sq;sqrt=a.sqrt;start=a.start;stop=a.stop;stroke=a.stroke;strokeCap=a.strokeCap;strokeJoin=a.strokeJoin;strokeWeight=
    a.strokeWeight;subset=a.subset;tan=a.tan;text=a.text;textAlign=a.textAlign;textAscent=a.textAscent;textDescent=a.textDescent;textFont=a.textFont;textLeading=a.textLeading;textMode=a.textMode;textSize=a.textSize;textWidth=a.textWidth;texture=a.texture;textureMode=a.textureMode;tint=a.tint;translate=a.translate;triangle=a.triangle;trim=a.trim;unbinary=a.unbinary;unhex=a.unhex;updatePixels=a.updatePixels;vertex=a.vertex;year=a.year;this.__defineGetter__("displayHeight",function(){return a.displayHeight});
    this.__defineGetter__("displayWidth",function(){return a.displayWidth});this.__defineGetter__("frameCount",function(){return a.frameCount});this.__defineGetter__("frameRate",function(){return a.frameRate});this.__defineGetter__("height",function(){return a.height});this.__defineGetter__("key",function(){return a.key});this.__defineGetter__("keyCode",function(){return a.keyCode});this.__defineGetter__("keyPressed",function(){return a.keyPressed});this.__defineGetter__("mousePressed",function(){return a.mousePressed});
    this.__defineGetter__("mouseX",function(){return a.mouseX});this.__defineGetter__("mouseY",function(){return a.mouseY});this.__defineGetter__("online",function(){return!0});this.__defineGetter__("pixels",function(){return a.pixels});this.__defineGetter__("pmouseX",function(){return a.pmouseX});this.__defineGetter__("pmouseY",function(){return a.pmouseY});this.__defineGetter__("screenHeight",function(){return a.screenHeight});this.__defineGetter__("screenWidth",function(){return a.screenWidth});this.__defineGetter__("width",
    function(){return a.width})};



  SketchStatementscomments.name = 'SketchStatementscomments';

  function SketchStatementscomments() {}

  SketchStatementscomments.prototype.setup = function() {
    (function(processing){injectProcessingApi(processing);size=function csModeApiInjectIffy (){processing.size.apply(processing,arguments);injectProcessingApi(processing);}})(this);
    size(200, 200);
    return background(255);
  };

  return SketchStatementscomments;

})();
