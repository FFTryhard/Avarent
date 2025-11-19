#include <fstream>
#include <iostream>
int main(){

std::ofstream test("C:/Users/H/Desktop/C++ Tryharding/.csv files/test.csv");
if(!test.is_open()) std::cerr << "Fail\n";
else {
    test << "a,b,c\n1,2,3\n";
    test.close();
}
return 0;
}