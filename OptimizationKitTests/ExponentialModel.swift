//
//  ExponentialModel.swift
//  OptimizationKitTests
//
//  Created by Jonathan Birge on 3/14/22.
//

import OptimizationKit

/// Test by fitting common problem of exponential decay. This `Fittable` class generates `n` points of data following an exponential decay with amplitude 1 and decay rate 1. It sets the initial guess for both model parameters to be off by 20 percent, though this can be changed by varying `initparams`. This allows easily checking convergence for a variety of conditions.
class ExponentialDecayModel: Fittable {
    var x: [Double] = []
    var y: [Double] = []
    var n: Int
    var initparams: [Double] = [2, 2]

    private var res: [Double] = []

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
        for k in 0..<n {
            x.append(3*Double(k)/Double(n))
            y.append(evalfun(at:x[k], with:[1.0, 1.0]))
            res.append(0.0)
        }
    }

    func fitresiduals(for params: [Double]) -> [Double] {
        for k in 0..<x.count {
            res[k] = evalfun(at: x[k], with: params) - y[k]
        }
        return res
    }

    private func evalfun(at x: Double, with params: [Double]) -> Double {
        return params[0] * exp(-params[1] * x)
    }
}

class AnalyticExponentialDecayModel: ExponentialDecayModel, AnalyticFittable {
    func jacobian(at params: [Double]) -> [[Double]] {
        var Ja: [Double] = []
        var Jb: [Double] = []
        for kx in 0..<x.count {
            Ja.append( exp(-params[1]*x[kx]) )
            Jb.append( -params[0]*exp(-params[1]*x[kx])*x[kx] )
        }
        return [Ja, Jb]
    }
}

class NoisyExponentialDecayModel: ExponentialDecayModel {
    init() {
        let x0: [Double] = [0, 1, 2, 3, 4, 5, 6]
        let y0: [Double] = [1.047, 0.2864, 0.288, 0.07777, 0.121, -0.0001342, 0, 0.01]

        super.init(n: x0.count)

        // replace default perfect data with noisy data
        self.x = x0
        self.y = y0
    }
}
