using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace GradientMatrixFactorisation
{
	class VectorialImplementation
	{
		public const int movie_count = 3952 + 1;
		public const int user_count = 6040 + 1;
		public double epsilon = 0.005;
		public double global_average;

		public double[][] U, V;
		public double[] U_average, V_average;
		private Tuple<int, int, double>[] ratings;
		//private Tuple<int, int, int>[] testset;
		private Tuple<int, int>[] test;
		private int dimension;
		public VectorialImplementation(int dimension, Tuple<int, int, double>[] ratings, Tuple<int, int>[] test, 
			double global_average, double[] U_average, double[] V_average)
		{
			this.ratings = ratings;
			this.test = test;
			this.dimension = dimension;
			this.global_average = global_average;
			this.U_average = U_average;
			this.V_average = V_average;
			//this.testset = testset;
			Random random = new Random();

			//Initialize system with given dimension parameter
			U = new double[user_count][];
			for (int i = 0; i < U.Length; i++)
			{
				U[i] = Enumerable
					.Repeat(1.0d, dimension)
					.Select(x => random.NextDouble())//* Math.Sqrt(5))
					.ToArray();
			}

			V = new double[movie_count][];
			for (int i = 0; i < V.Length; i++)
			{
				V[i] = Enumerable
					.Repeat(1.0d, dimension)
					.Select(x => random.NextDouble())//* Math.Sqrt(5))
					.ToArray();
			}

		}

		public void iterateAlgorithm(double epsilon, double lambda = -1.0)
		{
			this.epsilon = epsilon;
			if (lambda <= 0)
				foreach (var item in ratings)
				{
					//Copying values, since array is a reference type
					double[] cur_u = new double[dimension];
					double[] cur_v = new double[dimension];
					for (int i = 0; i < dimension; i++)
					{
						cur_u[i] = U[item.Item1][i];
						cur_v[i] = V[item.Item2][i];
					}

					//The current estimation of the model
					double cur_value = dot(cur_u, cur_v);

					//The current error of the model
					double cur_diff = item.Item3 - cur_value;

					//Modify model
					for (int i = 0; i < dimension; i++)
					{
						//Weight modification with the value in dot product for better optimization
						U[item.Item1][i] += cur_diff * epsilon * cur_v[i];
						V[item.Item2][i] += cur_diff * epsilon * cur_u[i];
						//if (U[item.Item1][i] >= 100)
						//{
						//	Console.WriteLine("NaN");
						//}
					}

				}
			else
				foreach (var item in ratings)
				{
					//Copying values, since array is a reference type
					double[] cur_u = new double[dimension];
					double[] cur_v = new double[dimension];
					for (int i = 0; i < dimension; i++)
					{
						cur_u[i] = U[item.Item1][i];
						cur_v[i] = V[item.Item2][i];
					}

					//The current estimation of the model
					double cur_value = dot(cur_u, cur_v);

					//The current error of the model
					double cur_diff = item.Item3 - cur_value;

					//Modify model
					for (int i = 0; i < dimension; i++)
					{
						//Weight modification with the value in dot product for better optimization
						//Prevent overlearning by subtracting lambda at correction
						U[item.Item1][i] += cur_diff * epsilon * cur_v[i] - lambda * cur_u[i];
						V[item.Item2][i] += cur_diff * epsilon * cur_u[i] - lambda * cur_v[i];
					}
				}
		}

		public double dot(double[] curU, double[] curV)
		{
			double result = 0;
			for (int i = 0; i < dimension; i++)
			{
				result += curU[i] * curV[i];
			}
			return result;
		}

		public double squareSum(double[] value)
		{
			double result = 0;
			foreach (var item in value)
			{
				result += item * item;
			}
			return result;
		}

		public void showResults()
		{
			foreach (var item in test)
			{
				Console.WriteLine("{0} {1} -> {2}", item.Item1, item.Item2, dot(U[item.Item1], V[item.Item2]));
				Thread.Sleep(500);
			}
		}

		public double sqareSetError(Tuple<int, int, int>[] ratings)
		{
			double error = 0;
			foreach (var item in ratings)
			{

				//The current estimation of the model
				double cur_value = dot(U[item.Item1], V[item.Item2]);

				//Corrigate back, to original scale
				cur_value += global_average;
				cur_value += U_average[item.Item1];
				cur_value += V_average[item.Item2];

				//The current error of the model
				double cur_diff = item.Item3 - cur_value;

				//Square error
				error += (cur_diff * cur_diff);
			}

			return Math.Sqrt(error / ratings.Length);
		}
	}
}
