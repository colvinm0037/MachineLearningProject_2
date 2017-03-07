package com.micah.ml;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.FourPeaksEvaluationFunction;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.SingleCrossOver;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

public class FourPeaks {

    
    private static final int N = 500;    
    private static final int T = N / 5;
    private static NumberFormat formatter = new DecimalFormat("#0.00");
    
	public static void runFourPeaks() {
		
		
		System.out.println("Running Four Peaks Tests");
		
		int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
         
        
  //      runRHC(hcp, ef);
  //      runSA(hcp, ef);
        
         runGA(gap, pop, ef);
        // runMIMIC();
	        

		
	}
	
	
	private static void runRHC(HillClimbingProblem hcp, EvaluationFunction ef) {
		
		System.out.println("Beginning Randomized Hill Climbing");
		
		 // RHC over various iteration amounts
		System.out.println("\nTraining over iterations");
		System.out.println("Iterations, Optimal Value");
        int iterations = 10000;
        while (iterations <= 300000) {
        	RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
	        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iterations);
	        fit.train();
	        iterations += 10000;
	        System.out.println(iterations + "\t" + ef.value(rhc.getOptimal()));
        }

        //  What is the best it can do in under 30 seconds
        Queue<Double> q = new LinkedList<Double>();
        iterations = 10000;
        System.out.println("Finding the best results withing a time frame");
        System.out.println("Iterations\tTime Taken\tOptimal Value");
        long timeTaken = 0;
        while (timeTaken < 30000) {
        	long currentTime = System.currentTimeMillis();
        	RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
	        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iterations);
	        fit.train();
	        timeTaken = System.currentTimeMillis() - currentTime;

	        System.out.println(iterations + "\t" + ef.value(rhc.getOptimal()) + "\t" + timeTaken);
	        iterations += 10000;
	        double result = ef.value(rhc.getOptimal());
        	boolean convergence = true;
	        if (q.size() == 5) {
	        	q.remove();	        	
	        } 
	        q.add(result);
	        for (double d : q) {
	        	if (Math.abs(d - N) > 1) {
	        		convergence = false;
	        		break;
	        	}
	        }
	        
	        if (convergence) break;
        }
	}
	
	private static void runSA(HillClimbingProblem hcp, EvaluationFunction ef) {
		
		System.out.println("Beginning SA");
		
		// Various Cooling values
        double value = 0.02;
        int iterations = 200000;
        System.out.println("\nTraining over " + iterations + " iterations with increasing cooling");
		System.out.println("Cooling, Optimal Value");
        for (double cooling = .02; cooling < 1; cooling +=.02) {
	        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, value, hcp);
	        FixedIterationTrainer fit = new FixedIterationTrainer(sa, iterations);
	        fit.train();
	        System.out.println(formatter.format(cooling) + "\t" + ef.value(sa.getOptimal()));
	        value += .02;	        
        }

        // Various iteration amounts
        System.out.println("\nTraining over various iterations with .95 cooling");
		System.out.println("Iterations, Optimal Value");
        iterations = 10000;
        while (iterations <= 300000) {
	        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
	        FixedIterationTrainer fit = new FixedIterationTrainer(sa, iterations);
	        fit.train();
	        System.out.println(iterations + "\t" + ef.value(sa.getOptimal()));
	        iterations += 10000;
        }

        // How far can it get in 30 seconds
        iterations = 10000;
        System.out.println("\nFinding the best results withing a time frame");
        System.out.println("Iterations\tTime Taken\tOptimal Value");
        Queue<Double> q = new LinkedList<Double>();
        long timeTaken = 0;
        while (timeTaken < 30000) {
        	long currentTime = System.currentTimeMillis();
        	SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
	        FixedIterationTrainer fit = new FixedIterationTrainer(sa, iterations);
	        fit.train();
	        timeTaken = System.currentTimeMillis() - currentTime;

	        System.out.println(iterations + "\t" + ef.value(sa.getOptimal()) + "\t" + timeTaken);
	        iterations += 10000;
	        double result = ef.value(sa.getOptimal());
        	boolean convergence = true;
	        if (q.size() == 5) {
	        	q.remove();	        	
	        } 
	        q.add(result);
	        for (double d : q) {
	        	if (Math.abs(d - N) > 1) {
	        		convergence = false;
	        		break;
	        	}
	        }
	        
	        if (convergence) break;	        
        }
	}

	private static void runGA(GeneticAlgorithmProblem gap, ProbabilisticOptimizationProblem pop, EvaluationFunction ef) {
        
		System.out.println("Beginning GA");
		
        System.out.println("\nTraining over various iterations");
		System.out.println("Iterations, Optimal Value");
		int iterations = 10000;
        while (iterations <= 300000) {

	        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(500, 50, 50, gap);
	        FixedIterationTrainer fit = new FixedIterationTrainer(ga, iterations);
	        fit.train();
	        System.out.println(iterations + "\t" + ef.value(ga.getOptimal()));
	        iterations += 10000;
        }
    
	}
	
	private static void runMIMIC() {
//		MIMIC mimic = new MIMIC(200, 20, pop);
//	    FixedIterationTrainer fit = new FixedIterationTrainer(mimic, iterations);
//	    fit.train();
//	    System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
	    
	}
	
}
