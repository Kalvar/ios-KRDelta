//
//  KRDeltaMath.h
//  KRDelta
//
//  Created by kalvar_lin on 2016/12/6.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface KRDeltaMath : NSObject

- (NSArray *)multiplyMatrix:(NSArray *)_matrix byNumber:(double)_number;
- (NSArray *)plusMatrix:(NSArray *)_matrix anotherMatrix:(NSArray *)_anotherMatrix;
- (double)randomMax:(double)_maxValue min:(double)_minValue;

@end
