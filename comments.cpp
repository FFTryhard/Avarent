#include <iostream>
#include <C:\Users\H\Desktop\C++ Tryharding\eigen-3.4.0\eigen-3.4.0\Eigen\Dense>
#include <algorithm>
#include <vector>

using namespace Eigen;
using namespace std;

std::vector<double> cosineSimilarity(std::vector <double> A, std::vector <double> B);

double cosineSimilarity(Eigen::Map<Eigen::VectorXd> &A, Eigen::Map<Eigen::VectorXd> &B);

int main(){
                  //d d  
    Matrix <float, 2, 3> m11;
    Matrix <float, 3, 2> m22;

    m11 << 1, 2, 3,
          4, 5, 6;

    m22 << 10, 11,
          20, 21,
          30, 31;

    

    cout << m11 * m22 << endl;

    vector<vector<int>> x = {{10, 20, 30}, {40, 50, 60}, {70, 80, 90}, {100, 110, 120}, {130, 140, 150}};
    vector <int> labels = {0, 0, 2, 2, 1}; 

    unordered_map<int, vector<int>> clusters;

    for(int i = 0; i < x.size(); i++) {
        clusters[labels[i]].push_back(i);
    }

    for(auto j : clusters){
        cout << j.first << " ";  //first is the key
        for(auto i : j.second){   //second are the values within the keys

            cout << i << " "; 

        }

        cout << endl;
    }

    cout << clusters[0][1]; //first is the key. second is what value
     cout << endl;

    

    Matrix <double, 1, 2> m1 = {1, 2};
    Matrix <double, 1, 2> m2 = {2 ,3};
    Matrix <double, 1, 2> m3 = {3, 4};

    cout << "Mean: ";
    cout << endl;

    Matrix <double, 1, 2> m = (m1 + m2 + m3) / 3;

    //divide each point from the mean

    Matrix <double, 1, 2> x1 = m1 - m;
    Matrix <double, 1, 2> x2 = m2 - m;
    Matrix <double, 1, 2> x3 = m3 - m;

    // now square it and transpose

    Matrix <double, 2, 2> x_1 = x1.transpose() * x1;
    Matrix <double, 2, 2> x_2 = x2.transpose() * x2;
    Matrix <double, 2, 2> x_3 = x3.transpose() * x3;

    Matrix <double, 2, 2> sum = x_1 + x_2 + x_3;

    Matrix <double, 2, 2> sumend = sum / 3;

     cout << sumend;

     cout << endl;

     //MAKE SURE TO HAVE SAME TYPES AND LITERALLY FOLLOW THE FORMULA

     //HOMEWORK. HOW TO KNOW WHAT ORIENTATION (COLUMN OR ROW ETC)

     cout << m22.transpose();

     cout << endl;


     std::vector<double> a = {1.0, 2.0, 3.0};

     std::vector<double> b = {4.0, 5.0, 6.0};

     Eigen::Map<Eigen::VectorXd> eigen_vector_A(a.data(), a.size());            //convert 2 vectors in eigen vectors
     Eigen::Map<Eigen::VectorXd> eigen_vector_B(b.data(), b.size());            //ofc you got one, it was the same vector

     std::cout << cosineSimilarity(eigen_vector_A, eigen_vector_B) << std::endl;








    


       






    





    return 0;
}

// ENTROPIES

/*
    for(int y = 0; y < entropies.size(); y++){
        for(int x = 0; x < entropies[y].getEntropies().size(); x++){  //each object has a method to call the vector where data is stored

            // [y] gets each data stream
            std::cout << entropies[y].getEntropies()[x] << " at ";
            // [x] loops over the entropy values in the vector that getEntropies returns
            std::cout << stream_chunks[y].getTime(x) << "ms";  //each window gets entropy. so we want per window, aka x
            std::cout << std::endl;   //we took each stream within stream chunks by reference, so we get the time data thats been through. its the timechunking object
                                      //each stream has updated data, and has methods to return the data. like time too.
        }                             // [y] = what stream. [x] = each float in the vector thats returned. its synced with each entropy. since those were the chunks before
        std::cout << std::endl;
        std::cout << std::endl;
    }
 
    std::cout << std::endl;
*/


// STRESS VECTORS HORIZONTAL

/*
    for(int x = 0; x < stressvectors[0].size(); x++){
        for(int y = 0; y < stressvectors.size(); y++){
            std::cout << stressvectors[y][x] << " ";
        }
        std::cout << std::endl;
    }   
    std::cout << std::endl;
*/
double cosineSimilarity(Eigen::Map<Eigen::VectorXd> &A, Eigen::Map<Eigen::VectorXd> &B){ //takes 2 vectors

    double dot_product = A.dot(B);  // A · B  //does A . B
    double norm_A = A.norm();       // ||A||
    double norm_B = B.norm();       // ||B||  //length vector from origin. "length" of a vector.

    //dividing by the norm allows you to focus on the direction of the vectors (the angle between them), rather than their magnitudes.
    
    return dot_product / (norm_A * norm_B);
}