//
//  Optimization.swift
//  EcoChef
//
//  Created by Jonathan Birge on 7/17/17.
//  Copyright © 2017-2022 Birge & Fuller. All rights reserved.
//

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

/// Super class that must be subclassed to implement a regression.
public class Fitter {
    /// Provide feedback during regression iterations?
    public var verbose: Bool = false
    /// Relative tolerance before iteration is terminated
    public var reltol: Double = 0.0001
    /// Maximum number of iterations that are allowed
    public var maxiters: Int = 32
    var system: Fittable  // regression model
    var testparams: [Double]?
    private let fdrel: Double = 0.0001
    private var iters: Int = 0
    
    var iterations: Int {
        return iters
    }
    
    public init(with sys: Fittable) {
        self.system = sys
    }
    
    /// Function that **must be overridden** by a subclass or an error will be thrown. This is the function that performs the actual regression.
    public func fit() throws -> [Double] {
        throw OptimizationError.noFitOverrideDefined
    }
    
    /// Initialize fit. This function **must be called at the beginning of a fit**. Returns the initial parameters.
    public func initFit() -> [Double]? {
        if verbose {
            print("Fitter: Starting fit...")
        }
        iters = 0
        testparams = system.fitparams
        return testparams
    }
    
    /// Helper function to check for termination. This function *must* be called once per iteration. Returns `false` if the fitter should stop. Guaranteed to return true the first time called.
    func checkTerminate(params: [Double]) -> Bool {
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
    
    /// Finite difference matrix approximating Jacobian. Columns (vectors) representing points and rows (vector of vectors) representing parameters.
    func jacobian(at params:[Double]) throws -> [[Double]] {
        var J: [[Double]] = []
        for kp in 0..<params.count {
            let dp = params[kp] * fdrel
            var params1 = params
            params1[kp] = params1[kp] + dp
            let x0 = try residuals(at: params)
            let x1 = try residuals(at: params1)
            J.append((x1 - x0)/dp)
        }
        return J
    }
    
    /// Jacobian function that takes works directly with Matrix objects.
    func jacobian(at params:Matrix<Double>) throws -> Matrix<Double> {
        let Jdata = try jacobian(at: params[column:0])
        return Matrix<Double>(Jdata)
    }
    
    func residuals(at params:[Double]) throws -> [Double] {
        return try system.fitresiduals(for: params)
    }
    
    func residuals(at beta:Matrix<Double>) throws -> Matrix<Double> {
        let betarray = beta.grid
        let residarray: [Double] = try residuals(at: betarray)
        return Matrix<Double>(residarray)
    }
}

/// Example `Fitter` class that implements a Gauss-Newton method for nonlinear regression. Works by solving the least squares problem using the Jacobian pseudo-inverse. Note how little code is required to implement a complete non-linear regression algorithm.
public class GaussNewtonFitter : Fitter {
    
    /// Implement Guass-Newton iterations, returning the current test vector after one step.
    public override func fit() throws -> [Double] {
        guard let initparams = initFit() else {
            throw OptimizationError.failedInit
        }
        var beta = Matrix<Double>(initparams)
        repeat {
            let r = try residuals(at: beta)
            let Jt = try jacobian(at: beta)  // transpose of Jacobian
            let Jpi = try inv(Jt * transpose(Jt)) * Jt  // pseudo-inverse
            beta = beta - Jpi * r
        } while checkTerminate(params: beta[column:0])
        
        return beta[column:0]
    }
}
