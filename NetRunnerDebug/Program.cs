using System;
using NetRunner;
namespace NetRunnerDebug
{
    class Program
    {
        static void Main(string[] args)
        {
            Net net = new Net(2, 2, 1, 2);
            net.RandomGenes();
            net.inputs = new double[2] { 1, 1 };
            net.In2Out();
            for(int i = 0; i < net.paths.Count; i++)
            {
                for(int j = 0; j < net.paths[i].Count; j++)
                {
                    Console.Write(net.paths[i][j]+" ");
                }
                Console.WriteLine("");
            }
            Console.WriteLine("");
            for (int i = 0; i < net.synapses.Count; i++)
            {
                for (int j = 0; j < net.synapses[i].Count; j++)
                {
                    Console.WriteLine("");
                    for (int k = 0; k < net.synapses[i][j].Count; k++)
                    {
                        Console.Write(net.synapses[i][j][k] + " ");
                    }
                }
                Console.WriteLine("");
            }
            Console.WriteLine("");
            for (int i = 0; i < net.weights.Count; i++)
            {
                for (int j = 0; j < net.weights[i].Count; j++)
                {
                    Console.WriteLine("");
                    for (int k = 0; k < net.weights[i][j].Count; k++)
                    {
                        Console.Write(net.weights[i][j][k] + " ");
                    }
                }
                Console.WriteLine("");
            }
            Console.WriteLine("");
            for (int i = 0; i < net.outputs.Length; i++)
            {
                Console.Write(net.outputs[i] + " ");
            }
            Console.ReadKey();
        }
    }
}
