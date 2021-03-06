//
//  Optimization.swift
//  EcoChef
//
//  Created by Jonathan Birge on 7/17/17.
//  Copyright © 2017-2022 Birge & Fuller. All rights reserved.
//

import Foundation

public enum OptimizationError: Error {
    case undefinedResidual
    case singularJacobian
    case didNotConverge
    case failedInit
    case noFitOverrideDefined
}

/// Protocol implemented by an object that provides a regression model.
public protocol Fittable {
    /// Returns the dimensionality of the parameter vector.
    var fitnparams: Int { get }
    /// Returns the number of data points over which we're fitting.
    var fitnpoints: Int { get }
    /// Returns the *starting* parameters for the fit. This is does not have to be updated during iterations.
    var fitparams: [Double] { get }
    /// Function that returns a vector of residuals given an array of test parameters `params`.
    func fitresiduals(for params:[Double]) throws -> [Double]
}

/// Interface for `Fittable` model that can produce analytic Jacobian matrices.
public protocol AnalyticFittable: Fittable {
    /// Returns Jacobian
    func jacobian(at params:[Double]) -> [[Double]]
}

/// Interface for delegate class that refines parameters in a regression iteration.
public protocol RegressionIterator {
    /// Used by `RegressionController` to inform `RegressionIterator` of parent system
    func setsystem(_ system: RegressionController)
    func refineparams(_ params: [Double]) throws -> [Double]
}

// TODO: Have RegressionController instantiate Iterator by passing Type
// TODO: Add parameter fit mask to allow for parameters to be "held"
/// Superclass for all regression implementations. This superclass doesn't actually implement any regression function, but provides a common interface and helper functions to simplify implementation of regression models, which can often be implemented with only a few lines of code.
public class RegressionController {
    /// Provide feedback during regression iterations?
    public var verbose: Bool = false
    /// Relative tolerance before iteration is terminated
    public var reltol: Double = 0.0001
    /// Maximum number of iterations that are allowed
    public var maxiters: Int = 32
    /// Relative offset for finite differences used to approximate derivatives
    public var fdrel: Double = 0.0001

    /// Regression system model
    var system: Fittable
    /// Regression iteration delegate
    var fitter: RegressionIterator

    var iterations: Int {
        return iters
    }

    private var testparams: [Double]?
    private var iters: Int = 0
    
    public init(for system: Fittable, using method: RegressionIterator) {
        self.system = system
        self.fitter = method
        method.setsystem(self)
    }
    
    /// Perform regression.
    public func regression() throws -> [Double] {
        guard var params = initFit() else {
            throw OptimizationError.failedInit
        }
        repeat {
            params = try fitter.refineparams(params)
        } while checkTerminate(params: params)

        return params
    }
    
    /// Initialize fit. This function **must be called at the beginning of a fit**. Returns the initial fit parameters.
    private func initFit() -> [Double]? {
        if verbose {
            print("Fitter: Starting fit...")
        }
        iters = 0
        testparams = system.fitparams
        return testparams
    }
    
    /// Check for termination. This function **must be called once per iteration**. Returns `true` if iterations should continue. Guaranteed to return `true` the first time called.
    private func checkTerminate(params: [Double]) -> Bool {
        // check iteration count
        if iters > maxiters {
            if verbose {
                print("Fitter: Max iterations reached.")
            }
            return false  //
        }
        
        // check tolerance
        if let lastparams = testparams {
            var delta: Double = 0
            var mag: Double = 0
            for kp in 0..<params.count {
                delta += pow(lastparams[kp] - params[kp], 2)
                mag += pow(params[kp], 2)
            }
            delta = sqrt(delta)
            mag = sqrt(mag)
            let relerr = delta/mag
            if verbose {
                print("Fitter: iter = \(iters), reltol = \(relerr)")
            }
            if relerr < reltol {
                if verbose {
                    print("Fitter: reltol reached.")
                }
                return false
            }
        }
        
        // keep going!
        iters += 1
        testparams = params
        return true
    }

    /// Compute Jacobian matrix. Will check to see if system model conforms to AnalyticFittable prototype. If so, will query it for the Jacobian. If not, will compute the Jacobian using finite differences.
    func jacobian(at params:[Double]) throws -> [[Double]] {
        if let analyticSystem = system as? AnalyticFittable {
            return analyticSystem.jacobian(at: params)
        } else {  // system not conformant to AnalyticFittable
            return try fdjacobian(at: params)
        }
    }

    /// Jacobian function that works directly with `Matrix<Double>` objects.
    func jacobian(at params:Matrix<Double>) throws -> Matrix<Double> {
        let Jdata = try jacobian(at: params[column:0])
        return Matrix<Double>(Jdata)
    }

    /// Finite difference approximate Jacobian. Columns (vectors) representing eval points and rows (vector of vectors) representing parameters. Uses central differences.
    private func fdjacobian(at params:[Double]) throws -> [[Double]] {
        var J: [[Double]] = []
        var params0 = params
        var params1 = params
        for kp in 0..<params.count {
            let dp = params[kp] * fdrel
            params0[kp] -= dp
            params1[kp] += dp
            let x0 = try system.fitresiduals(for: params0)
            let x1 = try system.fitresiduals(for: params1)
            J.append((x1 - x0)/(2*dp))
            params0[kp] = params[kp]
            params1[kp] = params[kp]
        }
        return J
    }

    /// Utility function to compute residuals.
    func residuals(at params:[Double]) throws -> [Double] {
        return try system.fitresiduals(for: params)
    }

    /// Utility function to compute residuals in form of `Matrix<Double>`
    func residuals(at beta:Matrix<Double>) throws -> Matrix<Double> {
        let betarray = beta.grid
        let residarray: [Double] = try system.fitresiduals(for: betarray)
        return Matrix<Double>(residarray)
    }
}

/// A `RegressionIterator` class that implements a Gauss-Newton method for nonlinear regression. Works by solving the least squares problem using the Jacobian pseudo-inverse. Note how little code is required to implement a complete non-linear regression algorithm.
public class GaussNewtonFitter : RegressionIterator {
    var system: RegressionController!

    public init() { }

    public func setsystem(_ system: RegressionController) {
        self.system = system
    }

    /// Implement Guass-Newton iterations, returning the current test vector after one step.
    public func refineparams(_ params: [Double]) throws -> [Double] {
        var beta = Matrix<Double>(params)
        let r = try system.residuals(at: beta)
        let Jt = try system.jacobian(at: beta)  // transpose of Jacobian
        let Jpi = try inv(Jt * transpose(Jt)) * Jt  // pseudo-inverse
        beta = beta - Jpi * r
        
        return beta[column:0]
    }
}
