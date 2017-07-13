/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 1000;

	default_random_engine gen;

	// Create distributions centered about incoming initial 'gps' coordinates
	normal_distribution<double> N_x(x, std[0]);
	normal_distribution<double> N_y(y, std[1]);
	normal_distribution<double> N_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++)
	{
		Particle p;
		p.id = i;

		if (isDebug)
		{
			p.x = x;
			p.y = y;
			p.theta = theta;
		}
		else
		{
			p.x = N_x(gen);
			p.y = N_y(gen);
			p.theta = N_theta(gen);
		}

		particles.push_back(p);
	}

	is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Pre-compute commonly used variables.
	double vt = velocity * delta_t;
	double vy = velocity / yaw_rate;
	double ydt = yaw_rate * delta_t;

	for (int i = 0; i < num_particles; i++)
	{
		double new_x, new_y, new_theta;
		Particle &p = particles[i];

		if (yaw_rate == 0)
		{
			new_x = p.x + vt * cos(p.theta);
			new_y = p.y + vt * sin(p.theta);
			new_theta = p.theta;
		}
		else
		{
			new_x = p.x + vy * ( sin(p.theta + ydt) - sin(p.theta) );
			new_y = p.y + vy * (cos(p.theta) - cos(p.theta + ydt));
			new_theta = p.theta + ydt;
		}


		if (isDebug){
			p.x = new_x;
			p.y = new_y;
			p.theta = new_theta;
		}
		else
		{
			// Generate new distributions centered around the new values.
			normal_distribution<double> N_x(new_x, std_pos[0]);
			normal_distribution<double> N_y(new_y, std_pos[1]);
			normal_distribution<double> N_theta(new_theta, std_pos[2]);
			default_random_engine gen;

			p.x = N_x(gen);
			p.y = N_y(gen);
			p.theta = N_theta(gen);
		}

	}

}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	if (isDebug)
		return;

	for (int j = 0; j < num_particles; j++)
	{
		Particle p = particles[j];
		// Set particle weight to 1 to initialize for taking the product of multiple weights.
		p.weight = 1;
		vector<double> sense_x;
		vector<double> sense_y;
		vector<int> associations;

		// Transform the car observations to this particle's perspective, and
		// get in world coordinates.
		for (int i = 0; i < observations.size(); i++) {
			LandmarkObs trans_obs;
			LandmarkObs obs;
			Map::single_landmark_s closest_landmark;
			obs = observations[i];

			// Space transformation from vehicle to particle (in map coords).
			trans_obs.x = p.x + (obs.x * cos(p.theta) - obs.y * sin(p.theta));
			trans_obs.y = p.y + (obs.x * sin(p.theta) + obs.y * cos(p.theta));

			// Find the nearest landmark for this observation.
			double closest_dist = sensor_range;
			int association = 0;
			int closest_id = 0;

			// Loop over LANDMARKS
			for (int k = 0; k < map_landmarks.landmark_list.size(); k++)
			{
				auto landmark = map_landmarks.landmark_list[k];
				double land_x = landmark.x_f;
				double land_y = landmark.y_f;

				// Calculate distance from the observation to each landmark.
				double calc_dist = sqrt( pow(trans_obs.x - land_x, 2.0) +
						                 pow(trans_obs.y - land_y, 2.0) );

				// Store this distance and landmark ID if closer.
				if (calc_dist < closest_dist) {
					closest_dist = calc_dist;
                    closest_id = landmark.id_i;
					closest_landmark = landmark;
				}
			} // loop landmarks


			// At this point, we have found the closest landmark for this observation.
			associations.push_back(closest_id);
			sense_x.push_back(trans_obs.x);
			sense_y.push_back(trans_obs.y);
			p = SetAssociations(p, associations, sense_x, sense_y);

			// Update the weight for this particle by adding the probability contribution
			// of this measurement / closest landmark pair.
			double mu_x = closest_landmark.x_f;
			double mu_y = closest_landmark.y_f;
			double c = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
			double c1 = 2 * std_landmark[0] * std_landmark[0];
			double c2 = 2 * std_landmark[1] * std_landmark[1];
			double t1 = pow(trans_obs.x - mu_x, 2.0)/c1;
			double t2 = pow(trans_obs.y - mu_y, 2.0)/c2;
			double val = t1 + t2;

			long double multiplier = c * exp(-val);

			// / Add probability contribution.
			if (multiplier > 0)
				p.weight *= multiplier;

		} // loop observations

		weights.push_back(p.weight);
		particles[j] = p;

		//cout << "Particle id=" << p.id << "  Weight=" << p.weight << endl;
	} // particles
}


void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	if (isDebug)
		return;

	default_random_engine gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end());
	vector<Particle> resample_particles;

	for (int i = 0; i < num_particles; i++)
	{
		resample_particles.push_back(particles[distribution(gen)]);
	}

	particles = resample_particles;
	weights.clear();
}


Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
