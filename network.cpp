#include <iostream>
#include <vector>
#include <chrono>

#include <armadillo>

#include "dataloader.cpp"

using namespace std;

void print_shape(string name, arma::mat x)
{
  cout << name << ' ' << x.n_cols << "x" << x.n_rows << endl;
}

template<typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n)
{
  auto first = v.cbegin() + m;
  auto last = v.cbegin() + n + 1;
  std::vector<T> vec(first, last);
  return vec;
}

double MSE(arma::mat y_pred, arma::mat y_true)
{
  return 0.5*arma::accu(arma::square(y_pred - y_true));
}

arma::mat MSE_prime(arma::mat y_pred, arma::mat y_true)
{
  return (y_pred - y_true);
}

arma::mat sigmoid(arma::mat x)
{
  return 1.0 / (1.0 + arma::exp(-x));
}

arma::mat sigmoid_prime(arma::mat x)
{
  auto s = sigmoid(x);
  return (1.0 - s) % s; // elementwise multiplication
}

class Dense
{
  public:
    arma::mat weights;
    arma::mat bias;
    arma::mat z;
    arma::mat grad_w;
    arma::mat grad_b;

    Dense(int units_in, int units_out);

    arma::mat forward(arma::mat x);
    void zero_grad();
    void update_weights(double lr, int batch_size);

};

Dense::Dense(int units_in, int units_out)
{
  weights = arma::mat(units_out, units_in, arma::fill::randn);
  bias = arma::mat(units_out, 1, arma::fill::zeros);
  zero_grad();
}

arma::mat Dense::forward(arma::mat x)
{
  z = (weights * x) + bias;
  arma::mat a = sigmoid(z);
  return a;
}

void Dense::zero_grad()
{
  grad_w = arma::zeros(arma::size(weights));
  grad_b = arma::zeros(arma::size(bias));

}

void Dense::update_weights(double lr, int batch_size)
{
  //  cout << arma::accu(grad_w) << endl;
  weights -= lr/batch_size * grad_w;
  bias -= lr/batch_size * grad_b;
}

class NN
{
  public:

    int input_size;
    vector<int> hidden_units;
    int num_layers;
    vector<Dense> layers;

    vector<arma::mat> activations;

    NN(int _input_size, vector<int> _hidden_units);
    arma::mat forward(arma::mat x);
    void backward(arma::mat delta);
    void update_weights(double lr, int batch_size);
    void zero_grad();

    void fit(vector<arma::mat> x_train, vector<arma::mat> y_train,
             vector<arma::mat> x_val, vector<arma::mat> y_val,
             double lr, int batch_size, int epochs);

};

NN::NN(int _input_size, vector<int> _hidden_units)
{
  input_size = _input_size;
  hidden_units = _hidden_units;
  num_layers = hidden_units.size();

  layers.push_back(Dense(input_size, hidden_units[0]));

  for (int l=1; l < num_layers; l++) {
    layers.push_back(Dense(hidden_units[l-1], hidden_units[l]));
  }
}

arma::mat NN::forward(arma::mat x)
{
  activations.clear();
  activations.push_back(x);
  for (int l=0; l < num_layers; l++) {
    x = layers[l].forward(x);
    activations.push_back(x);
  }
  return x;
}

void NN::backward(arma::mat delta)
{
  // output layer
  // cout << "output layer" << endl;
  delta = delta % sigmoid_prime(layers[num_layers-1].z); // element-wise multiplication
  layers[num_layers-1].grad_b = delta;
  // cout << "delta times a" << endl;
  layers[num_layers-1].grad_w = delta * activations[num_layers - 1].t();
  // cout << "hidden layers" << endl;
  // hidden layers
  for (int l=num_layers-2; l > -1; l--) {
    // cout << "hidden layers" << l << endl;
    auto sp = sigmoid_prime(layers[l].z);
    delta = (layers[l+1].weights.t() * delta) % sp;
    layers[l].grad_b = delta;
    layers[l].grad_w = delta * activations[l].t();
  }
}

void NN::update_weights(double lr, int batch_size)
{
  for (int l=0; l < num_layers; l++) {
    layers[l].update_weights(lr, batch_size);
  }
}

void NN::zero_grad()
{
  for (int l=0; l < num_layers; l++) {
    layers[l].zero_grad();
  }
}

void NN::fit(vector<arma::mat> x_train, vector<arma::mat> y_train,
             vector<arma::mat> x_val, vector<arma::mat> y_val,
             double lr, int batch_size, int epochs)
{
  int steps = x_train.size() / batch_size;
  for (int epoch=0; epoch < epochs; epoch++) {
    double loss_epoch = 0;
    double acc_epoch = 0;
    auto start = chrono::high_resolution_clock::now();
    for (int batch=0; batch < steps; batch++) {
      auto x_batch = slice(x_train, batch*batch_size, (batch+1)*batch_size);
      auto y_batch = slice(y_train, batch*batch_size, (batch+1)*batch_size);

      double loss = 0.0;
      double corr_count = 0.0;
      for (int k=0; k < x_batch.size(); k++) {
        auto x = x_batch[k];
        auto y = y_batch[k];

        auto y_pred = forward(x);
        loss += MSE(y_pred, y) / batch_size;
        auto grad_loss = MSE_prime(y_pred, y);
        zero_grad();
        backward(grad_loss);
        update_weights(lr, batch_size);
        corr_count += y_pred.index_max() == y.index_max();
      } // samples
      double acc = corr_count / batch_size;

      loss_epoch += loss / steps;
      acc_epoch += acc / steps;

    } // batches

    // validation
    double val_loss = 0.0;
    double corr_count = 0.0;
    for (int i=0; i < x_val.size(); i++) {
      auto x = x_val[i];
      auto y = y_val[i];
      auto y_p = forward(x);
      val_loss += MSE(y_p, y);
      corr_count += y_p.index_max() == y.index_max();
    } // validation
    val_loss = val_loss / x_val.size();
    double val_acc = corr_count / x_val.size();

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(stop - start);
    cout << "Epoch " << epoch+1 << "/" << epochs << " " << duration.count() <<
    "s loss " << loss_epoch << " acc " << 100*acc_epoch << "% val_loss " <<
    val_loss << " val_acc " << 100*val_acc << "%" << endl;
  } // epochs
}

int main()
{
  cout << "Loading train data" << endl;
  // 60000
  Dataloader train_data = Dataloader("dataset/train", 60000);
  cout << "Loading test data" << endl;
  // 10000
  Dataloader test_data = Dataloader("dataset/test", 10000);

  NN net = NN(28*28, {64, 32, 10});

  net.fit(train_data.x, train_data.y, test_data.x, test_data.y, 3, 64, 10);
  return 0;
}