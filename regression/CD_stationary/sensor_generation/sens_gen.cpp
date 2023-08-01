#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
using namespace std;


int main(){

    // input choices
    const int UNIFORM = 0;
    const int RANDOM  = 1;
    int choice;
    int n;
    double dx;

    // message
    cout << "\n---SENSOR GENERATION ROUTINE---" << endl;
    cout <<  "//////////////////////////////\n" << endl;

    // output file
    ofstream outfile("sensor_points.dat");
    if (!outfile){
        cerr << "Unable to open file.\n" << endl;
        return 1;
    }

    // uniform [0] or random [1]
    cout << "Choose [0] for uniform or [1] for random:  ";
    cin  >> choice;
    cout << "\n" << endl;

    // uniform case
    if (choice == UNIFORM){

        // ask for number of sensors
        //int n;
        cout << "Generating n^2 uniform sensors, choose n: ";
        cin  >> n;
        cout << "\n" << endl;

        // validate input
        if (n <= 0){
            cerr << "INVALID INPUT: terminating program\n" << endl;
            return 1;
        }

        // generate discretized unit interval
        double dx = 1.0/(n + 1);
        double grid[n];
        for (int i = 0; i <= n - 1; ++i){
            grid[i] = dx * (i + 1);
        }

        // add locations to output file
        outfile << fixed << setprecision(2);
        for (int i = 0; i <= n - 1; ++i){
            for (int j = 0; j <= n - 1; ++j){
                outfile << setw(4) << setfill('0') << grid[i] << " " 
                        << setw(4) << setfill('0') << grid[j] << endl;
            }
        }        
    }

    // random cases
    if (choice == RANDOM){

        // ask for number of sensors
        int n;
        cout << "Generating n random sensors, choose n: ";
        cin  >> n;
        cout << "\n" << endl;

        // validate input
        if (n <= 0){
            cerr << "INVALID INPUT: terminating program\n" << endl;
            return 1;
        }

        // initialize random device
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(0.0, 1.0);

        // add locations to output file
        for (int i = 0; i <= n - 1; ++i){
            outfile << fixed << setprecision(2) << dis(gen) << " "
                    << fixed << setprecision(2) << dis(gen) << endl;
        }
    }

    // message
    cout << "---SENSOR GENERATION COMPLETE---" << endl;
    cout << "////////////////////////////////\n" << endl;

    outfile.close();
    return 0;
}