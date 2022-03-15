//
//  OptimizationTests.swift
//  EcoChef
//
//  Created by Jonathan Birge on 7/18/17.
//  Copyright © 2017 Birge Clocks. All rights reserved.
//

import XCTest
@testable import OptimizationKit

class OptimizationTests: XCTestCase {
    var noiseTestModel = NoisyExponentialDecayModel()
    var largeTestModel = ExponentialDecayModel(n: 1024)
    var hugeTestModel = ExponentialDecayModel(n: 1024*16)
    var analyticTestModel = AnalyticExponentialDecayModel(n: 1024*16)

    func assertDefaultParams(_ params: [Double]) {
        XCTAssertEqual(params[0], 1.0, accuracy: 0.05)
        XCTAssertEqual(params[1], 1.0, accuracy: 0.05)
    }

    func testNoisyGaussNewtonFit() {
        let fitter = GaussNewtonFitter(with: noiseTestModel)
        fitter.verbose = true
        fitter.reltol = 0.00001
        do {
            let p: [Double] = try fitter.fit()
            XCTAssertEqual(p[0], 1.02, accuracy: 0.05)
            XCTAssertEqual(p[1], 0.9, accuracy: 0.05)
        } catch {
            XCTFail("GaussNewton failed on small noise test")
        }
    }

    func testLargeGaussNewtonFit() {
        let fitter = GaussNewtonFitter(with: hugeTestModel)
        fitter.verbose = true
        fitter.reltol = 0.00001
        do {
            let p: [Double] = try fitter.fit()
            assertDefaultParams(p)
        } catch {
            XCTFail("GaussNewton failed on large scale test")
        }
    }

    func testAnalyticGaussNewtonFit() {
        let fitter = GaussNewtonFitter(with: analyticTestModel)
        fitter.verbose = true
        fitter.reltol = 0.00001
        do {
            let p: [Double] = try fitter.fit()
            assertDefaultParams(p)
        } catch {
            XCTFail("GaussNewton failed on analytic model test")
        }
    }
    
    func testGaussNewtonPerformance() {
        let measOptions = XCTMeasureOptions.default
        measOptions.iterationCount = 32
        let fitter = GaussNewtonFitter(with: largeTestModel)
        fitter.reltol = 0.00001
        measure(options: measOptions) {
            var p: [Double]
            do {
                p = try fitter.fit()
                assertDefaultParams(p)
            } catch {
                XCTFail("GaussNewton failed on performance test")
            }
        }
    }
    
}
