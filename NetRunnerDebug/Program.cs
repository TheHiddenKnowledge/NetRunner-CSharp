using NetRunner;
using System;
using MathNet.Numerics.LinearAlgebra;
namespace NetRunnerDebug
{
    class Program
    {
        static void Main(string[] args)
        {
            Net net = new Net(2, 2, new int[] { 2 }, .01, .9, 3);
            double[] input = new double[] { 1, 1 };
            net.inputs = Vector<double>.Build.DenseOfArray(input);
            net.In2Out();
            for (int i = 0; i < net.weights.Count; i++)
            {
                for (int j = 0; j < net.weights[i].RowCount; j++)
                {
                    Console.WriteLine("");
                    for (int k = 0; k < net.weights[i].ColumnCount; k++)
                    {
                        Console.Write(net.weights[i][j, k] + " ");
                    }
                }
                Console.WriteLine("");
            }
            Console.WriteLine("");
        }
    }
}
