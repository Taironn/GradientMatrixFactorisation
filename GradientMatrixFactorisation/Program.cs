using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace GradientMatrixFactorisation
{
	class Program
	{
		public const int movie_count = 3952 + 1;
		public const int user_count = 6040 + 1;
		public static double[] U, V;
		public static Tuple<int, int, int>[] ratings;
		public static Tuple<int, int, int>[] testingset;
		public static Tuple<int, int>[] test;
		public static double epsilon = 0.1;
		static void Main(string[] args)
		{

			#region InputHandling

			string[] data = System.IO.File.ReadAllLines("./moviedata/ratings.train");

			//Contains all ratings in UserID, MovieID, Rating form
			int i = 0;
			int j = (int)(data.Length * 0.2);
			ratings = new Tuple<int, int, int>[j];
			testingset = new Tuple<int, int, int>[data.Length-j];
			//foreach (var item in data)
			//{
			//	string[] splitted = item.Split(' ');
			//	ratings[i++] = new Tuple<int, int, int>(int.Parse(splitted[0]), int.Parse(splitted[1]), int.Parse(splitted[2]));
			//}
			for (int k = 0; k < j; k++)
			{
				string[] splitted = data[k].Split(' ');
				ratings[k] = new Tuple<int, int, int>(int.Parse(splitted[0]), int.Parse(splitted[1]), int.Parse(splitted[2]));
			}
			for (int k = j; k < data.Length; k++)
			{
				string[] splitted = data[k].Split(' ');
				testingset[k-j] = new Tuple<int, int, int>(int.Parse(splitted[0]), int.Parse(splitted[1]), int.Parse(splitted[2]));
			}

			data = System.IO.File.ReadAllLines("./moviedata/ratings.test");

			//Contains all test data in UserID, MovieID form
			test = new Tuple<int, int>[data.Length];
			i = 0;
			foreach (var item in data)
			{
				string[] splitted = item.Split(' ');
				test[i++] = new Tuple<int, int>(int.Parse(splitted[0]), int.Parse(splitted[1]));
			}
			#endregion

			//Basic implementation here
			//basicImplementation();

			//Vectorial implementation, the dimension for the vectors in U and V is the first parameter
			VectorialImplementation syst = new VectorialImplementation(20, ratings, test);
			for (int k = 0; k < 500; k++)
			{
				syst.iterateAlgorithm(0.0025);
				Console.WriteLine("{0} iterations done.", k + 1);
				Console.WriteLine("Current error on learning set is: {0}", syst.sqareSetError(ratings));
				Console.WriteLine("Current error on testing  set is: {0}", syst.sqareSetError(testingset));
			}

			//Console.WriteLine("Current error is: {0}", syst.sqareSetError());

			//Show calculated values
			//syst.showResults();

			Console.ReadKey();
		}

		static void basicImplementation()
		{
			Random random = new Random();
			U = Enumerable
				.Repeat(1.0f, user_count)
				.Select(x => random.NextDouble() * Math.Sqrt(5))
				.ToArray();

			V = Enumerable
				.Repeat(1.0f, movie_count)
				.Select(x => random.NextDouble() * Math.Sqrt(5))
				.ToArray();

			//Iterate the algorithm
			for (int j = 0; j < 5; j++)
			{
				GradientIteration();
				Console.WriteLine("{0} iterations done.", j + 1);
			}

			//Write predicition of test data
			foreach (var item in test)
			{
				Console.WriteLine("{0} {1} -> {2}", item.Item1, item.Item2, U[item.Item1] * V[item.Item2]);
				Thread.Sleep(500);
			}

			Console.ReadKey();
		}

		static void GradientIteration()
		{
			foreach (var item in ratings)
			{
				//The current estimation of the model
				double cur_value = U[item.Item1] * V[item.Item2];

				//The current error of the model
				double cur_diff = item.Item3 - cur_value;

				//Update model according to gradient
				U[item.Item1] += epsilon * cur_diff;
				V[item.Item2] += epsilon * cur_diff;
			}
		}
	}
}
