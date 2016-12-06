//
//  KRDeltaFetcher.m
//  KRDelta
//
//  Created by kalvar_lin on 2016/12/6.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRDeltaFetcher.h"
#import "KRDelta.h"

@implementation KRDeltaFetcher

+ (instancetype)sharedFetcher
{
    static dispatch_once_t pred;
    static KRDeltaFetcher *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRDeltaFetcher alloc] init];
    });
    return _object;
}

- (instancetype)init
{
    self = [super init];
    if( self )
    {
        
    }
    return self;
}

- (void)save:(KRDelta *)object forKey:(NSString *)key
{
    if( object && key )
    {
        [[NSUserDefaults standardUserDefaults] setObject:[NSKeyedArchiver archivedDataWithRootObject:object] forKey:key];
        [[NSUserDefaults standardUserDefaults] synchronize];
    }
}

- (void)removeForKey:(NSString *)key
{
    if( key )
    {
        [[NSUserDefaults standardUserDefaults] removeObjectForKey:key];
        [[NSUserDefaults standardUserDefaults] synchronize];
    }
}

// Fetching saved object that all recoreded parameters from trained network.
- (KRDelta *)objectForKey:(NSString *)key
{
    if( key )
    {
        NSData *objectData = [[NSUserDefaults standardUserDefaults] valueForKey:key];
        return objectData ? [NSKeyedUnarchiver unarchiveObjectWithData:objectData] : nil;
    }
    return nil;
}

@end
