using NetRunner;
using System;
using MathNet.Numerics.LinearAlgebra;
namespace NetRunnerDebug
{
    class Program
    {
        static void Main(string[] args)
        {
            Net net = new Net(2, 1, new int[] { 1 }, .01, .9, 3);
            double[] input = new double[] { 1, 1 };
            net.inputs = Vector<double>.Build.DenseOfArray(input);
            net.outputs = net.In2Out();
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
            for (int i = 0; i < net.biases.Count; i++)
            {
                Console.WriteLine("");
                    for (int k = 0; k < net.biases[i].Count; k++)
                    {
                        Console.Write(net.biases[i][k] + " ");
                    }
            }
            Console.WriteLine("");
            for (int i = 0; i < net.layerout.Count; i++)
            {
                Console.WriteLine("");
                for (int j = 0; j < net.layerout[i].Count; j++)
                {
                    Console.Write(net.layerout[i][j]+" ");
                }
            }
            Console.WriteLine("");
            for (int i = 0; i < net.layersquish.Count; i++)
            {
                Console.WriteLine("");
                for (int j = 0; j < net.layersquish[i].Count; j++)
                {
                    Console.Write(net.layersquish[i][j] + " ");
                }
            }
            Console.WriteLine("");
            Console.ReadLine();
        }
    }
}
