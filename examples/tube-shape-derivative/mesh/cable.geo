r1 = 1.0;
r2 = 0.1;
cl1 = 0.5;
cl2 = 1.0;

x0_off = -0.3;
x1_off = 0.3;

Point(0) = {0, 0, 0, cl1};
Point(1) = {-r1, 0, 0, cl1};
Point(2) = {r1, 0, 0, cl1};
Point(3) = {0, r1, 0, cl1};
Point(4) = {0, -r1, 0, cl1};

Point(5) = {x0_off, x1_off, 0, cl2};
Point(6) = {-r2+x0_off, x1_off, 0, cl2};
Point(7) = {r2+x0_off, x1_off, 0, cl2};
Point(8) = {x0_off, r2+x1_off, 0, cl2};
Point(9) = {x0_off, -r2+x1_off, 0, cl2};

Circle(1) = {2, 0, 3};
Circle(2) = {3, 0, 1};
Circle(3) = {1, 0, 4};
Circle(4) = {4, 0, 2};
Circle(5) = {8, 5, 6};
Circle(6) = {6, 5, 9};
Circle(7) = {9, 5, 7};
Circle(8) = {7, 5, 8};
Line Loop(9) = {2, 3, 4, 1};
Line Loop(10) = {8, 5, 6, 7};
Plane Surface(11) = {9, 10};
Physical Line(1) = {3, 4, 1, 2};
Physical Line(2) = {8, 5, 6, 7};
Physical Surface(0) = {11};
