#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // If this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // If this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Initial state vector
  x_ = VectorXd(5);
  x_.fill(0);  // Initialization

  // Initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;


  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2.0;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  // Initialization flag
  is_initialized_ = false;

  // Set state dimension
  n_x_ = 5;

  // Set augmented dimension
  n_aug_ = 7;

  // Define spreading parameter
  lambda_ = 3 - n_x_;

  Xsig_pred_ = MatrixXd(n_x_, (2 * n_aug_) + 1);
  Xsig_pred_.fill(0); // Initialization


  // Weights of sigma points 
  weights_ = VectorXd((2 * n_aug_) + 1);
  weights_.fill(0); // Initialization

  // Set weights
  weights_(0) = lambda_ / (lambda_ + n_aug_); // Initial
  for (int i = 1 ; i < (2 * n_aug_ + 1); ++i)
  {
    // All the others
    weights_(i) = 1 / (2 * (lambda_ + n_aug_));
  }

  // Time initialization
  prev_time_ = 0.0;

}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) 
{
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  std::cout << "Start measurement " << std::endl;

  if (!is_initialized_) 
  {

    std::cout << "Kalman Filter Initialization " << std::endl;

    // Sensor type checking
    if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      // Set the state with the initial location and zero velocity
      x_ << meas_package.raw_measurements_[0], 
            meas_package.raw_measurements_[1], 
            0,
            0,
            0; 
    }

    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      // Set the initial state with radar measurements
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double rho_dot = meas_package.raw_measurements_[2];

      x_ << (rho * cos(phi)), 
            (rho * sin(phi)), 
            0, 
            0, 
            0;
    }

    else
    {
      std::cout << "Unknown sensor at initialization" << std::endl; 
    }

    prev_time_ = meas_package.timestamp_;
    is_initialized_ = true;

    return;
  }

  // Compute dt between the current and previous measurements in seconds
  double dt = (meas_package.timestamp_ - prev_time_) / 1000000.0;
  prev_time_ = meas_package.timestamp_;

  std::cout << "Prediction" << std::endl; 
  // Prediction step
  Prediction(dt);

  // Measurement update
  if (meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    std::cout << "Update Lidar" << std::endl; 
    UpdateLidar(meas_package);
    std::cout << "-------------" << std::endl; 
  }

  else if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    std::cout << "Update Radar" << std::endl; 
    UpdateRadar(meas_package); 
    std::cout << "-------------" << std::endl; 
  }

}

void UKF::Prediction(double delta_t)
{
  /**
   * TODO: Complete this function! Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */

  /**
   * In the prediction step, we rely purely on our mmotion model
   * sensor measurement does not play any role at this point
   * */

  /**
   * Sigma points will be calculated in the augmented way
   * This to represent the uncertainty of the covariance matrix,
   * for that reason we used the augmente state (7 dimension).
   * */

  // Create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // Create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // Create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // Augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0.0;
  x_aug(6) = 0.0;

  // Augmented covariance matrix
  P_aug.fill(0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  MatrixXd P_sqrt = P_aug.llt().matrixL();

  // Augmented sigma points calculation
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_ ; ++i)
  {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * P_sqrt.col(i);
    Xsig_aug.col(i + n_aug_ + 1) = x_aug - sqrt(lambda_ + n_aug_) * P_sqrt.col(i);
  }

  /**
   * At this step we simply put every sigma points
   * into the process model
   * */
  // Sigma points predictions - This is not the state yet
  for (int i = 0; i < (2 * n_aug_) + 1; ++i)
  {
    // Value extraction from the sigma points
    double px = Xsig_aug(0, i);
    double py = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yaw_rate = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yaw_rate = Xsig_aug(6, i);

    double px_pred = 0.0;
    double py_pred = 0.0;

    // Avoid division by zero
    if (fabs(yaw_rate) > 0.001)
    {
      /* Cornering */
      px_pred = px + ((v / yaw_rate) * (sin(yaw + yaw_rate * delta_t) - sin(yaw))) + (0.5 * delta_t * delta_t * cos(yaw) * nu_a);
      py_pred = py + ((v / yaw_rate) * (-cos(yaw + yaw_rate * delta_t) + cos(yaw))) + (0.5 * delta_t * delta_t * sin(yaw) * nu_a);
    }

    else
    {
      /* There is no yaw rate */
      px_pred = px + (v * cos(yaw) * delta_t) + (0.5 * delta_t * delta_t * cos(yaw) * nu_a);
      py_pred = py + (v * sin(yaw) * delta_t) + (0.5 * delta_t * delta_t * sin(yaw) * nu_a);
    }

    // std::cout << "************************************************" << std::endl;
    // std::cout << "px_pred: " << px_pred << std::endl;
    // std::cout << "py_pred: " << py_pred << std::endl;

    double v_pred = v + (delta_t * nu_a);
    double yaw_pred =  yaw + (yaw_rate * delta_t) + (0.5 * delta_t * delta_t * nu_yaw_rate);
    double yaw_rate_pred = yaw_rate + (delta_t * nu_yaw_rate);

    // Predicted sigma points 
    Xsig_pred_(0, i) = px_pred;
    Xsig_pred_(1, i) = py_pred;
    Xsig_pred_(2, i) = v_pred;
    Xsig_pred_(3, i) = yaw_pred;
    Xsig_pred_(4, i) = yaw_rate_pred;
  }

  // Create augmented mean vector
  VectorXd x_pred = VectorXd(n_x_);
  x_pred.fill(0);
  MatrixXd P_pred = MatrixXd(n_x_, n_x_);
  P_pred.fill(0);

  // State prediction and state covariance matrix prediction
  // Predicted mean state
  for (int i = 0; i < (2 * n_aug_) + 1; ++i)
  {
    x_pred += (weights_(i) * Xsig_pred_.col(i));
  }

  // Predicted state covariance matrix 
  for (int i = 0; i < (2 * n_aug_) + 1; ++i)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_pred;
    // Angle normalization
    while (x_diff(3)> M_PI){ x_diff(3) -= 2.*M_PI; }
    while (x_diff(3)< -M_PI){ x_diff(3) += 2.*M_PI; }

    P_pred = P_pred + (weights_(i) * x_diff * x_diff.transpose());
  }

  x_ = x_pred;
  P_ = P_pred;
}


void UKF::UpdateRadar(MeasurementPackage meas_package) 
{
  /**
   * TODO: Complete this function! Use radar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  int n_z = 3;
  MatrixXd Zsig = MatrixXd::Zero(n_z, 2 * n_aug_ + 1);

  // predict measurements------------------------  

  for (int i = 0; i < (2 * n_aug_ + 1); ++i)
  {
    // Transform sigma points into measurement space
    double px = Xsig_pred_(0, i);  
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);
    double yaw_rate = Xsig_pred_(4, i);
      
    // Measurement model
    Zsig(0, i) = sqrt((px * px) + (py * py));    
    Zsig(1, i) = atan2(py, px);  
    Zsig(2, i) = ((px * cos(yaw) * v) + (py * sin(yaw) * v)) / sqrt((px * px) + (py * py));  
  }

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);

  // Calculate mean predicted measurement
  for (int i = 0; i < (2 * n_aug_ + 1); ++i)
  {
    z_pred = z_pred + (weights_(i) * Zsig.col(i));     
  }

  // Calculate innovation covariance matrix S
  for (int i = 0; i < (2 * n_aug_ + 1); ++i)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    // Angle normalization
    while (z_diff(1)> M_PI){ z_diff(1) -= 2.*M_PI; }
    while (z_diff(1)<-M_PI){ z_diff(1) += 2.*M_PI; }
    
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_ * std_radr_, 0.0, 0.0,
       0.0, std_radphi_ * std_radphi_, 0.0, 
       0.0, 0.0, std_radrd_ * std_radrd_;
  
  S = S + R;

  // State updating
  MatrixXd Tc = MatrixXd(n_x_, n_z); // State dimension X Measurement dimension 
  Tc.fill(0);

  for (int i = 0; i < (2 * n_aug_ + 1); ++i)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalization
    while (x_diff(3)> M_PI){ x_diff(3) -= 2.*M_PI; }
    while (x_diff(3)< -M_PI){ x_diff(3) += 2.*M_PI; }

    VectorXd z_diff = Zsig.col(i) - z_pred;
    // Angle normalization
    while (z_diff(1)> M_PI){ z_diff(1) -= 2.*M_PI; }
    while (z_diff(1)< -M_PI){ z_diff(1) += 2.*M_PI; }

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  VectorXd z = meas_package.raw_measurements_; // Incoming measurement 

  VectorXd z_diff = z - z_pred; 
  while(z_diff(1) > M_PI){ z_diff(1) -= 2.*M_PI; } 
  while(z_diff(1) < -M_PI){ z_diff(1) += 2.*M_PI; }

  // Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}


void UKF::UpdateLidar(MeasurementPackage meas_package) 
{
  /**
   * TODO: Complete this function! Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  int n_z = 2;

  // Create matrix for sigma points in measurement space: 3x15
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);

  /* Measurement prediction */
  for (int i = 0; i < (2 * n_aug_ + 1); ++i)
  {
    // Transform sigma points into measurement space
    double px = Xsig_pred_(0, i);  
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);
    double yaw_rate = Xsig_pred_(4, i);
      
    // Measurement model
    Zsig(0, i) = px;    
    Zsig(1, i) = py; 
  }

  // Calculate mean predicted measurement
  for (int i = 0; i < (2 * n_aug_ + 1); ++i)
  {
    z_pred = z_pred + (weights_(i) * Zsig.col(i));     
  }

  // Calculate innovation covariance matrix S
  for (int i = 0; i < (2 * n_aug_ + 1); ++i)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  
  MatrixXd R = MatrixXd(n_z, n_z);
  R <<  std_laspx_ * std_laspx_, 0.0,
        0.0, std_laspy_ * std_laspy_; 

  S = S + R;

  // State updating
  MatrixXd Tc = MatrixXd(n_x_, n_z); // State dimension X Measurement dimension 
  Tc.fill(0);

  for (int i = 0; i < (2 * n_aug_) + 1; ++i)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalization
    while (x_diff(3)> M_PI){ x_diff(3) -= 2.*M_PI; }
    while (x_diff(3)< -M_PI){ x_diff(3) += 2.*M_PI; }

    VectorXd z_diff = Zsig.col(i) - z_pred;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  VectorXd z = meas_package.raw_measurements_; // Incoming measurement 

  VectorXd z_diff = z - z_pred; 

  // Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}
