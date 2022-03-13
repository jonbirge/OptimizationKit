//
//  OptimizationTests.swift
//  EcoChef
//
//  Created by Jonathan Birge on 7/18/17.
//  Copyright © 2017 Birge Clocks. All rights reserved.
//

import XCTest
@testable import OptimizationKit

/// Test by fitting common problem of exponential decay
class GenExponentialTest: Fittable {
    var x: [Double] = []
    var y: [Double] = []
    var n: Int
    
    var fitnparams: Int {
        return 2
    }
    
    var fitnpoints: Int {
        return x.count
    }
    
    // Memoryless Fittable model
    var fitparams: [Double] {
        return [0.9, 0.9]
    }
    
    init(n: Int) {
        self.n = n
        for k in 0...n-1 {
            self.x.append(Double(k)/Double(n))
            self.y.append(evalfun(at:x[k], with:[1.0, 1.0]))
        }
    }
    
    func evalfun(at x: Double, with params: [Double]) -> Double {
        return params[0] * exp(-params[1] * x)
    }
    
    func fitresiduals(for params: [Double]) -> [Double] {
        var res: [Double] = []
        for k in 0...(x.count - 1) {
            res.append(evalfun(at:x[k], with:params) - y[k])
        }
        return res
    }
}

class NoisyExponentialTest: GenExponentialTest {
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
    var funtest = NoisyExponentialTest()
    var gentest = GenExponentialTest(n: 256)
    
    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testGaussNewtonFit() {
        let fitter = GaussNewtonFitter(with: gentest)
        fitter.verbose = true
        do {
            let p: [Double] = try fitter.fit()
            XCTAssertEqual(p[0], 1, accuracy: 0.01)
            XCTAssertEqual(p[1], 1, accuracy: 0.01)
        } catch {
            XCTFail("GaussNewton threw up")
        }
    }
    
    func testGaussNewtonPerformance() {
        let m = 64
        let fitter = GaussNewtonFitter(with: funtest)
        self.measure {
            var p: [Double]
            do {
                for k in 1...m {
                    let tc0 = Double(k)/Double(m) + 0.5
                    fitter.setInitial(params: [0.5, tc0])
                    p = try fitter.fit()
                    XCTAssertEqual(p[0], 1.0, accuracy: 0.01)
                    XCTAssertEqual(p[1], 1.0, accuracy: 0.01)
                }
            } catch {
                XCTFail("GaussNewton threw up")
            }
        }
    }
    
}
