lc1 = 0.02;
lc2 = lc1/2;
cx = 0.3;
cy = 0.5;
a = 0.05;
b = 0.15000000000000002;
width = 3*lc1;
L = 1.0;
H = 1.0;

// Outer shell
Point(1) = {0,0,0,lc1};
Point(2) = {L, 0, 0, lc1};
Point(3) = {L, H, 0, lc1};
Point(4) = {0, H, 0, lc1};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};


// Inner geo
// Inner circle
Point(6) = {cx,cy,0,lc2};
Point(7) = {cx-b, cy, 0, lc2};
Point(8) = {cx+b, cy, 0, lc2};
Point(9) = {cx, cy+a, 0, lc2};

Point(10) = {cx-b-width, cy, 0, lc1};
Point(11) = {cx, cy+a+width, 0, lc1};
Point(12) = {cx+b+width, cy, 0, lc1};

Line(6) = {10,7};
Ellipse(7) = {7,6,7,9};
Ellipse(8) = {9,6,9,8};

Line(9) = {8,12};
Ellipse(10) = {12,6,12,11};
Ellipse(11) = {11,6,11,10};

Line Loop(10) = {6,7,8,9,10,11};
Line Loop(5) = {1,2,3,4};

Plane Surface(12) = {10};

Symmetry {0,1,0,-cy}{Duplicata{Surface{12};}}
Line Loop(11) = {6,15,16,9,18,19};
Plane Surface(14) = {5,10,11};


//Physical stuff
Physical Surface(13) = {12,14};
Physical Surface(14) = {13,14};

bc = 1;
object = 2;

Physical Line(bc) = {1,2,3,4};
Physical Line(object) = {15,16,8,7};


