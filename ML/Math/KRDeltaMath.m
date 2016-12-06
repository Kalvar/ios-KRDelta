//
//  KRDeltaMath.m
//  KRDelta
//
//  Created by kalvar_lin on 2016/12/6.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRDeltaMath.h"

@implementation KRDeltaMath

- (instancetype)init
{
    self = [super init];
    if( self )
    {
        
    }
    return self;
}

// ex : 0.5f * [1, 2]
- (NSArray *)multiplyMatrix:(NSArray *)_matrix byNumber:(double)_number
{
    NSMutableArray *_array = [NSMutableArray new];
    for( NSNumber *_value in _matrix )
    {
        double _newValue = _number * [_value doubleValue];
        [_array addObject:[NSNumber numberWithDouble:_newValue]];
    }
    return _array;
}

// ex : [1, 2] + [3, 4]
- (NSArray *)plusMatrix:(NSArray *)_matrix anotherMatrix:(NSArray *)_anotherMatrix
{
    NSMutableArray *_array = [NSMutableArray new];
    NSInteger _index       = 0;
    for( NSNumber *_value in _matrix )
    {
        double _newValue = [_value doubleValue] + [[_anotherMatrix objectAtIndex:_index] doubleValue];
        [_array addObject:[NSNumber numberWithDouble:_newValue]];
        ++_index;
    }
    return _array;
}

- (double)randomMax:(double)_maxValue min:(double)_minValue
{
    return ((double)arc4random() / ( RAND_MAX * 2.0f ) ) * (_maxValue - _minValue) + _minValue;
}

@end
