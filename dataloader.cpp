#include <iostream>
#include <vector>
#include <armadillo>

using namespace std;

class Dataloader
{
  public:
    vector<arma::mat> x;
    vector<arma::mat> y;
    string path;
    int num_files;
    Dataloader(string _path, int _num_files);

};

Dataloader::Dataloader(string _path, int _num_files)
{
  path = _path;
  num_files = _num_files;

  arma::mat current_x;
  arma::mat current_y;
  char current_path[50];
  for (int i=1; i < num_files+1; i++)
  {
    sprintf(current_path, (path + "/x/%06d.txt").c_str(), i);
    current_x.load(current_path, arma::raw_ascii);
    x.push_back(current_x.t());
    sprintf(current_path, (path + "/y/%06d.txt").c_str(), i);
    current_y.load(current_path, arma::raw_ascii);
    y.push_back(current_y.t());
  }

}