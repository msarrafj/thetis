lc=1e12;
fac=15000;
R1=0.45*fac;
R2=fac*100;
Point(1) = {R1, 0, 0,  lc};
Point(2) = {-R1,0, 0,  lc};
Point(3) = {0, -R1, 0, lc};
Point(4) = {0, R1, 0,  lc};
Point(5) = {0, 0, 0, lc};
Point(6) = {R2, 0, 0,  lc};
Point(7) = {-R2, 0, 0, lc};
Point(8) = {0, -R2, 0, lc};
Point(9) = {0, R2, 0,  lc};
Circle(1) = {2,5,4};
Circle(2) = {4,5,1};
Circle(3) = {1,5,3};
Circle(4) = {3,5,2};
Circle(5) = {7, 5, 9};
Circle(6) = {9, 5, 6};
Circle(7) = {6, 5, 8};
Circle(8) = {8, 5, 7};
Line Loop(9) = {2, 3, 4, 1};
Plane Surface(10) = {9};
Line Loop(11) = {6, 7, 8, 5};
Plane Surface(12) = {11, 9};
Field[11] = MathEval;
Field[11].F = "500"; // LC in the resolved circle
Field[1] = MathEval;
Field[1].F = "Sqrt(x*x+y*y)";
Field[2] = MathEval;
Field[2].F = "15000*0.45";// Radius of the resolved circle
Field[3] = Max;
Field[3].FieldsList = {1, 2};
Field[12] = MathEval;
Field[12].F = "F3/15000/0.45";
Field[4] = MathEval;
Field[4].F = "((F12*F12*F12-1)*1+1)*F11";
Field[5] = MathEval;
Field[5].F = "F1/2";// A third of the radius
Field[6] = Min;
Field[6].FieldsList = {4,5};
Field[7] = Max;
Field[7].FieldsList = {6,11};
Background Field = 7;
Physical Line(13) = {2, 3, 4, 1};
Physical Line(14) = {6, 5, 8, 7};
Physical Surface(1) = {10};
Physical Surface(2) = {12};
Mesh.Algorithm = 6; // frontal=6, delannay=5, meshadapt=1
