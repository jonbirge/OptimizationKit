//
//  OptimizationTests.swift
//  EcoChef
//
//  Created by Jonathan Birge on 7/18/17.
//  Copyright © 2017 Birge Clocks. All rights reserved.
//

import XCTest
@testable import OptimizationKit

/// Test by fitting common problem of exponential decay. This `Fittable` class generates `n` points of data following an exponential decay with amplitude 1 and decay rate 1. It sets the initial guess for both model parameters to be off by 20 percent, though this can be changed by varying `initparams`. This allows easily checking convergence for a variety of conditions.
class ExponentialDecayModel: Fittable {
    var x: [Double] = []
    var y: [Double] = []
    var n: Int
    var initparams: [Double] = [0.5, 0.5]
    
    var fitnparams: Int {
        return 2
    }
    
    var fitnpoints: Int {
        return x.count
    }
    
    var fitparams: [Double] {
        return initparams
    }
    
    init(n: Int) {
        self.n = n
        for k in 0...n-1 {
            self.x.append(Double(k)/Double(n))
            self.y.append(evalfun(at:x[k], with:[1.0, 1.0]))
        }
    }
    
    // TODO: pre-allocate array to x.count
    func fitresiduals(for params: [Double]) -> [Double] {
        var res: [Double] = []
        for k in 0..<x.count {
            res.append(evalfun(at:x[k], with:params) - y[k])
        }
        return res
    }
    
    private func evalfun(at x: Double, with params: [Double]) -> Double {
        return params[0] * exp(-params[1] * x)
    }
}

class NoisyExponentialTest: ExponentialDecayModel {
    init() {
        let x0: [Double] = [0, 1, 2, 3, 4, 5, 6]
        let y0: [Double] = [1.047, 0.2864, 0.288, 0.07777, 0.121, -0.0001342, 0, 0.01]
        
        super.init(n: x0.count)
        
        // replace default perfect data with noisy data
        self.x = x0
        self.y = y0
    }
}

class OptimizationTests: XCTestCase {
    var noiseTestModel = NoisyExponentialTest()
    var largeTestModel = ExponentialDecayModel(n: 1024)
    var hugeTestModel = ExponentialDecayModel(n: 10000)
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testGaussNewtonFit() {
        let fitter = GaussNewtonFitter(with: noiseTestModel)
        fitter.verbose = true
        fitter.reltol = 0.00001
        do {
            let p: [Double] = try fitter.fit()
            XCTAssertEqual(p[0], 1.02, accuracy: 0.05)
            XCTAssertEqual(p[1], 0.9, accuracy: 0.05)
        } catch {
            XCTFail("GaussNewton failed on small test")
        }
    }

    func testLargeGaussNewtonFit() {
        let fitter = GaussNewtonFitter(with: hugeTestModel)
        fitter.verbose = true
        fitter.reltol = 0.00001
        do {
            let p: [Double] = try fitter.fit()
            XCTAssertEqual(p[0], 1.0, accuracy: 0.05)
            XCTAssertEqual(p[1], 1.0, accuracy: 0.05)
        } catch {
            XCTFail("GaussNewton failed on small test")
        }
    }
    
    func testGaussNewtonPerformance() {
        let measOptions = XCTMeasureOptions.default
        measOptions.iterationCount = 32
        let fitter = GaussNewtonFitter(with: largeTestModel)
        measure(options: measOptions) {
            var p: [Double]
            do {
                p = try fitter.fit()
                XCTAssertEqual(p[0], 1.0, accuracy: 0.01)
                XCTAssertEqual(p[1], 1.0, accuracy: 0.01)
            } catch {
                XCTFail("GaussNewton failed on large test")
            }
        }
    }
    
}
