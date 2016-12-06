//
//  KRDeltaActivation.h
//  KRDelta
//
//  Created by kalvar_lin on 2016/12/6.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface KRDeltaActivation : NSObject

- (double)tanh:(double)x slope:(double)slope;
- (double)sigmoid:(double)x slope:(double)slope;
- (float)sgn:(double)x;
- (double)rbf:(double)x sigma:(double)sigma;

- (double)partialTanh:(double)x slope:(double)slope;
- (double)partialSigmoid:(double)x slope:(double)slope;
- (double)partialRBF:(double)x sigma:(double)sigma;
- (double)partialSgn:(double)x;

@end
