//
//  KRDeltaFetcher.h
//  KRDelta
//
//  Created by kalvar_lin on 2016/12/6.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@class KRDelta;

@interface KRDeltaFetcher : NSObject

+ (instancetype)sharedFetcher;
- (instancetype)init;

- (void)save:(KRDelta *)object forKey:(NSString *)key;
- (void)removeForKey:(NSString *)key;
- (KRDelta *)objectForKey:(NSString *)key;

@end
