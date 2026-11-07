// Generative scene shader. Every concept the networks detect is a procedural
// layer whose opacity/intensity is the (smoothed) concept weight; mood
// (valence/arousal) grades colour and animation speed; audio features add
// reactivity. Layers are composited back-to-front so mixtures of concepts
// ("a storm over the city at night") give coherent combined scenes.

export const VERT = `#version 300 es
in vec2 p; out vec2 uv;
void main(){ uv = p*0.5+0.5; gl_Position = vec4(p,0.,1.); }`;

export const FRAG = `#version 300 es
precision highp float;
in vec2 uv; out vec4 o;
uniform vec2 R; uniform float t; uniform float C[22];
uniform float val; uniform float aro; uniform vec4 au; uniform float Q; uniform float flash;
#define OCEAN C[0]
#define BEACH C[1]
#define FOREST C[2]
#define MOUNTAIN C[3]
#define DESERT C[4]
#define CITY C[5]
#define SPACE C[6]
#define NIGHT C[7]
#define RAIN C[8]
#define SNOW C[9]
#define FIRE C[10]
#define SUNSET C[11]
#define RIVER C[12]
#define FLOWERS C[13]
#define STORM C[14]
#define UNDERWATER C[15]
#define TECH C[16]
#define MUSIC C[17]
#define LOVE C[18]
#define CLOUDS C[19]
#define BIRDS C[20]
#define FIELDS C[21]

float h1(float n){ return fract(sin(n)*43758.5453); }
float h2(vec2 p){ return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
float noise(vec2 p){ vec2 i=floor(p), f=fract(p); f=f*f*(3.-2.*f);
  return mix(mix(h2(i),h2(i+vec2(1,0)),f.x), mix(h2(i+vec2(0,1)),h2(i+vec2(1,1)),f.x), f.y); }
float fbm(vec2 p){ float s=0., a=.5; for(int i=0;i<6;i++){ if(float(i)>=Q) break; s+=a*noise(p); p=p*2.03+17.1; a*=.5; } return s; }
vec3 over(vec3 c, vec3 l, float a){ return mix(c,l,clamp(a,0.,1.)); }
float sat(float x){ return clamp(x,0.,1.); }
float heart(vec2 p){ p.y -= sqrt(abs(p.x))*.5; return length(p) - .5; }

void main(){
  float asp = R.x/R.y;
  vec2 q = vec2((uv.x-.5)*asp, uv.y);       // aspect-correct, y up, x centered
  float sp = .6 + .9*(aro*.5+.5);             // arousal -> tempo
  float T = t*sp;
  float dark = sat(max(NIGHT, SPACE*.9) + STORM*.5);
  float hor = .42 + .08*MOUNTAIN - .1*UNDERWATER;

  // ---- sky
  vec3 day = mix(vec3(.62,.80,.98), vec3(.22,.48,.88), uv.y);
  vec3 dusk = mix(vec3(1.,.55,.25), vec3(.35,.18,.45), sat((uv.y-hor)*1.6));
  vec3 nite = mix(vec3(.05,.07,.18), vec3(.0,.0,.03), uv.y);
  vec3 col = over(day, dusk, SUNSET);
  col = over(col, nite, dark);
  col = over(col, mix(vec3(.45,.47,.5), vec3(.22,.23,.27), uv.y), STORM*.8 + RAIN*.4);
  col = over(col, mix(vec3(.85,.78,.6), vec3(.6,.75,.95), uv.y), DESERT*.35);

  // ---- space: stars, nebula, planet
  float stars = step(.996, h2(floor(q*vec2(220.)))) * (.6+.4*sin(t*3.+h2(floor(q*220.))*40.));
  col += stars * sat(dark + SPACE) * vec3(1.) * step(hor, uv.y);
  if (SPACE > .01) {
    float n = fbm(q*2.5 + vec2(T*.03, 0.));
    vec3 neb = mix(vec3(.4,.1,.6), vec3(.1,.5,.9), fbm(q*4.-T*.02));
    col += neb * pow(n, 2.5) * SPACE * 1.2;
    vec2 pc = q - vec2(.45*asp*.6, .78);
    float pl = smoothstep(.13, .125, length(pc));
    col = over(col, mix(vec3(.9,.5,.3), vec3(.4,.2,.15), sat(pc.x*5.+.5)) * (.6+.4*noise(pc*20.)), pl*SPACE);
  }
  // ---- sun / moon
  vec2 sunp = vec2(-.3*asp*.5, mix(.75, hor+.05, SUNSET));
  float sd = length(q - sunp);
  vec3 sunc = mix(vec3(1.,.95,.7), vec3(1.,.45,.15), SUNSET);
  float sunA = (1.-dark) * sat(1.-STORM-RAIN*.6);
  col += sunc * (smoothstep(.09,.08,sd) + .35*exp(-sd*6.)) * sunA;
  float moon = smoothstep(.06,.055,sd) - smoothstep(.06,.055,length(q-sunp-vec2(.025,.012)));
  col = over(col, vec3(.95,.95,.85), moon * NIGHT * (1.-SPACE));

  // ---- clouds
  float cl = fbm(vec2(q.x*1.5 + T*.04, q.y*4.)) * smoothstep(hor, hor+.2, uv.y);
  float clA = sat(CLOUDS + STORM + RAIN*.6 + SNOW*.3);
  col = over(col, mix(vec3(1.), vec3(.3,.32,.38), sat(STORM+RAIN*.5)), smoothstep(.45,.8,cl)*clA);

  // ---- birds
  if (BIRDS > .01) for (int i=0;i<7;i++){
    float fi=float(i);
    vec2 bp = vec2(fract(h1(fi)+T*.05*(.6+h1(fi+3.)))*asp*1.4-asp*.7, .62+.25*h1(fi+7.) + .02*sin(T+fi));
    vec2 d = q-bp; float flap = .5+.5*sin(T*8.+fi);
    float b = abs(abs(d.x)*(.5+.6*flap) - d.y + abs(d.x)*.4) ;
    col = over(col, vec3(.08), smoothstep(.006,.0,b)*step(abs(d.x),.03)*BIRDS);
  }

  // ---- mountains
  if (MOUNTAIN > .01) {
    for (int l=0;l<2;l++){ float fl=float(l);
      float m = hor + .05 + (.25-.08*fl)*fbm(vec2(q.x*(1.2+fl)+fl*7., 1.)) ;
      vec3 mc = mix(vec3(.35,.4,.5), vec3(.2,.25,.3), fl);
      mc = mix(mc, vec3(.93), smoothstep(m-.04, m, uv.y)*sat(SNOW+.4)); // snow caps
      col = over(col, mc*(1.-dark*.7), step(uv.y, m)*MOUNTAIN);
    }
  }
  // ---- default ground (meadow / earth / snow), hidden in pure space scenes
  {
    float gn = noise(vec2(q.x*8., uv.y*30.));
    vec3 gc = mix(vec3(.28,.42,.22), vec3(.36,.32,.25), sat(DESERT+CITY+STORM*.5)) * (.85+.3*gn);
    gc = mix(gc, vec3(.9,.92,.96)*(.9+.1*gn), sat(SNOW*1.2));
    col = over(col, gc*(1.-dark*.75), step(uv.y, hor) * sat(1.-SPACE*1.5));
  }
  // ---- city skyline
  if (CITY > .01) {
    float bx = floor(q.x*14.); float bh = hor + .05 + .3*h1(bx) ;
    float inB = step(uv.y, bh) * step(hor - .12, uv.y);
    vec2 w = fract(vec2(q.x*14.*4., uv.y*60.));
    float lit = step(.78 - dark*.2, h2(floor(vec2(q.x*56., uv.y*60.)))) * step(.35,w.x)*step(w.x,.8)*step(.4,w.y)*step(w.y,.85);
    vec3 bc = mix(vec3(.45,.48,.55), vec3(.06,.07,.1), dark);
    bc += lit * vec3(1.,.85,.5) * (.25 + dark*.9);
    col = over(col, bc, inB*CITY);
  }
  // ---- ground: fields, desert dunes, forest
  float g = step(uv.y, hor);
  if (FIELDS > .01) {
    float hill = hor - .03 + .04*sin(q.x*3.+1.);
    vec3 fc = mix(vec3(.35,.6,.2), vec3(.85,.7,.3), .5+.5*sin(q.x*40.+uv.y*80.));
    col = over(col, fc*(1.-dark*.6), step(uv.y,hill)*FIELDS);
  }
  if (DESERT > .01) {
    float dune = hor - .02 + .05*sin(q.x*4. + 2.*fbm(q*3.)) ;
    vec3 dc = mix(vec3(.93,.75,.45), vec3(.75,.5,.28), sat((dune-uv.y)*6.));
    dc *= .85 + .15*sin(q.x*60.+uv.y*200.);
    col = over(col, dc*(1.-dark*.6), step(uv.y,dune)*DESERT);
  }
  if (FOREST > .01) {
    for (int l=0;l<3;l++){ float fl=float(l);
      float s = 10. + fl*6.; float x = q.x*s + fl*3.3; float id = floor(x);
      float tx = fract(x)-.5; float th = (.12 + .1*h1(id+fl*9.)) * (1.+fl*.3);
      float base = hor - .02 - fl*.06;
      float tri = step(abs(tx)*th*2.2, base + th - uv.y) * step(base - .02, uv.y);
      float trunk = step(abs(tx), .06) * step(base - .04, uv.y) * step(uv.y, base);
      vec3 tc = mix(vec3(.12,.35,.18), vec3(.05,.18,.1), fl/2.) * (1.-dark*.7);
      col = over(col, tc, sat(tri + trunk) * FOREST);
    }
  }
  // ---- ocean + beach + river
  if (OCEAN > .01) {
    float wv = .006*sin(q.x*30.+T*2.) + .004*sin(q.x*57.-T*3.);
    float sea = step(uv.y, hor + wv);
    vec3 oc = mix(vec3(.1,.45,.65), vec3(.02,.15,.3), sat((hor-uv.y)*2.5));
    float sp2 = pow(noise(vec2(q.x*20., uv.y*90. - T*.5)), 6.) * 2.;
    oc += sp2 * mix(vec3(1.), sunc, SUNSET) * (1.-dark*.6);
    col = over(col, oc*(1.-dark*.7), sea*OCEAN);
  }
  if (BEACH > .01) {
    float shore = .16 + .02*sin(q.x*5.+T*.6);
    vec3 sc = vec3(.95,.85,.62)*(1.-dark*.6);
    col = over(col, mix(sc, vec3(1.), smoothstep(shore-.01, shore, uv.y)*.6), step(uv.y, shore+.008)*BEACH);
    // palm
    vec2 pp = q - vec2(.55*asp*.6, .1);
    float trunkP = step(abs(pp.x + .03*pp.y*pp.y*8. ), .012)*step(0.,pp.y)*step(pp.y,.35);
    float ang = atan(pp.y-.35, pp.x); float r = length(pp-vec2(0.,.35));
    float leaf = step(r, .18)*step(.5, fract(ang*7./6.283 + .1*sin(T)))*step(r*.8, .12+.04*sin(ang*3.));
    col = over(col, vec3(.1,.25,.1)*(1.-dark*.5), sat(trunkP+leaf)*BEACH);
  }
  if (RIVER > .01) {
    float cx = .25*sin(uv.y*7.+.5) * (hor-uv.y)*2.;
    float wdt = .02 + (hor-uv.y)*.5;
    float rv = step(abs(q.x - cx), wdt) * step(uv.y, hor);
    vec3 rc = vec3(.25,.55,.75) + .3*pow(noise(vec2(q.x*30., uv.y*40.+T*2.)),4.);
    col = over(col, rc*(1.-dark*.6), rv*RIVER);
  }
  // ---- flowers
  if (FLOWERS > .01) {
    vec2 cell = vec2(q.x*18., uv.y*30.); vec2 id = floor(cell); vec2 f = fract(cell)-.5;
    f.x += .15*sin(T*1.5+id.y);
    float rnd = h2(id); float fl = smoothstep(.22,.18,length(f)) * step(.35, rnd) * step(uv.y, hor-.02);
    vec3 fc = .5 + .5*cos(6.283*(rnd + vec3(0.,.33,.67)));
    col = over(col, fc*(1.-dark*.5), fl*FLOWERS*sat(1.2-uv.y*2.));
  }
  // ---- fire
  if (FIRE > .01) {
    vec2 fp = vec2(q.x*3., uv.y*2. - T*1.2);
    float f = fbm(fp + fbm(fp*1.5)) ;
    float h = sat(1. - uv.y*(1.9-.6*au.x)) ;
    float flame = smoothstep(.35, .9, f*h*1.6);
    vec3 fc = mix(vec3(.9,.15,.0), vec3(1.,.85,.3), flame);
    col = over(col, fc, flame*FIRE);
    col += vec3(1.,.4,.1)*.25*FIRE*h;
  }
  // ---- underwater
  if (UNDERWATER > .01) {
    vec3 uc = mix(vec3(.0,.18,.32), vec3(.05,.55,.7), uv.y);
    vec2 cp = q*6.; float c = 0.;
    for (int i=0;i<3;i++){ cp += vec2(sin(cp.y+T*.7), cos(cp.x-T*.6)); c += .5*abs(sin(cp.x+cp.y)); }
    uc += vec3(.2,.5,.5)*pow(sat(c/3.), 4.);
    for (int i=0;i<8;i++){ float fi=float(i);
      vec2 bp = vec2((h1(fi)-.5)*asp, fract(h1(fi+4.) + T*.08*(1.+h1(fi))));
      uc += vec3(.6,.8,.9)*smoothstep(.012,.008,abs(length(q-bp)-.008)); }
    // fish
    for (int i=0;i<5;i++){ float fi=float(i);
      vec2 fp = q - vec2(fract(h1(fi+9.)+T*.04*(1.+fi*.2))*asp*1.4-asp*.7, .2+.5*h1(fi+2.));
      float body = length(fp*vec2(1.,2.2)) - .03; float tail = step(abs(fp.y), (fp.x+.03)*-.8)*step(fp.x,-.025)*step(-.06,fp.x);
      uc = over(uc, .5+.5*cos(6.283*(h1(fi)+vec3(0.,.3,.6))), sat(step(body,0.)+tail)); }
    col = over(col, uc, UNDERWATER);
  }
  // ---- technology: neon grid + data rain
  if (TECH > .01) {
    float gy = hor - uv.y;
    vec2 gp = vec2(q.x/(gy+.05), 1./(gy+.05) + T*2.);
    float grid = (smoothstep(.05,.0,abs(fract(gp.x)-.5)-.45) + smoothstep(.05,.0,abs(fract(gp.y)-.5)-.45)) * step(0.,gy);
    vec3 tc = vec3(.1,.9,1.)*grid*(.3+gy*2.);
    float colm = floor(q.x*40.);
    float drop = fract(h1(colm)*10. - T*(.3+h1(colm+1.)*.5));
    float glyph = step(.5, h2(floor(vec2(q.x*40., uv.y*50.)) + floor(T*6.)))*step(0.,uv.y-hor);
    tc += vec3(.2,1.,.5)*glyph*pow(sat(1.-abs(uv.y-1.+drop)*3.),3.)*.8;
    col = mix(col, col*.4 + tc, TECH);
  }
  // ---- music: audio-reactive equaliser + pulse rings
  if (MUSIC > .01) {
    float bi = floor(uv.x*32.);
    float lvl = .1 + .5*fract(sin(bi*12.9+floor(T*6.))*43.)*(.3+au.x*1.5) * (.5+.5*sin(T*4.+bi));
    vec3 mc = .5+.5*cos(6.283*(bi/32. + T*.1 + vec3(0.,.33,.67)));
    float bar = step(uv.y, lvl*.5)*step(.15, fract(uv.x*32.));
    float ring = smoothstep(.02,.0,abs(fract(length(q-vec2(0.,.5))*3. - T*.8)-.5)-.47);
    col = over(col, mc, (bar*.8 + ring*.25*(.4+au.y))*MUSIC);
  }
  // ---- love: floating hearts + pink glow
  if (LOVE > .01) {
    for (int i=0;i<9;i++){ float fi=float(i);
      vec2 hp = vec2((h1(fi*3.1)-.5)*asp*.9, fract(h1(fi)+T*.06*(1.+h1(fi+1.))));
      float s = .04 + .03*h1(fi+5.);
      float hd = heart((q-hp)/s*vec2(1.,-1.)*-1.);
      col = over(col, vec3(1.,.3,.5), smoothstep(.05,-.05,hd)*LOVE*.9);
    }
    col = mix(col, col*vec3(1.1,.85,.95)+vec3(.08,0.,.04), LOVE*.5);
  }
  // ---- weather on top: rain, snow, lightning
  if (RAIN > .01 || STORM > .01) {
    float r = sat(RAIN + STORM*.7);
    vec2 rp = vec2(q.x*80. + uv.y*8., uv.y*6. + T*7.);
    float drop = step(.92, h2(floor(rp))) * smoothstep(.0,.5,fract(rp.y));
    col = over(col, vec3(.75,.8,.9), drop*r*.6);
  }
  if (SNOW > .01) {
    for (int l=0;l<3;l++){ float fl=float(l)+1.;
      vec2 sp3 = q*12.*fl + vec2(sin(T*.5+fl)*.5, T*.8/fl*2.);
      vec2 f = fract(sp3)-.5; float id = h2(floor(sp3));
      col = over(col, vec3(1.), smoothstep(.08/fl+.02,.0,length(f+.3*(id-.5)))*step(.6,id)*SNOW);
    }
    col = mix(col, col + vec3(.15,.17,.2)*(1.-uv.y), SNOW*.4);
  }
  col += vec3(.8,.85,1.) * flash * STORM;

  // ---- mood grading: valence -> warmth/saturation, arousal -> contrast
  float lum = dot(col, vec3(.299,.587,.114));
  col = mix(vec3(lum), col, .65 + .5*sat(val*.5+.5));
  col *= mix(vec3(.9,.95,1.08), vec3(1.08,1.,.9), sat(val*.5+.5));
  col = mix(vec3(.5), col, 1. + .25*aro);
  col *= 1. + .25*au.w; // onset pulse
  float vig = 1. - .35*length(uv-.5);
  o = vec4(pow(max(col*vig, 0.), vec3(.95)), 1.);
}`;
