//
//  KRDeltaActivation.h
//  KRDelta
//
//  Created by kalvar_lin on 2016/12/6.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRDeltaPartial.h"

@interface KRDeltaActivation : NSObject

@property (nonatomic) KRDeltaPartial *partial;

- (double)tanh:(double)x slope:(double)slope;
- (double)sigmoid:(double)x slope:(double)slope;
- (double)sgn:(double)x;
- (double)rbf:(double)x sigma:(double)sigma;
- (double)reLU:(double)x;
- (double)leakyReLU:(double)x;
- (double)eLU:(double)x;

@end
