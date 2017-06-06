Pod::Spec.new do |s|
  s.name         = "KRDelta"
  s.version      = "1.5.1"
  s.summary      = "KRDelta is a supervised learning which Delta rule of Machine Learning."
  s.description  = <<-DESC
                   It still works in recommendation system and real-time user behavior analysis on mobile apps.
                   DESC
  s.homepage     = "https://github.com/Kalvar/ios-KRDelta"
  s.license      = { :type => 'MIT', :file => 'LICENSE' }
  s.author       = { "Kalvar Lin" => "ilovekalvar@gmail.com" }
  s.social_media_url = "https://twitter.com/ilovekalvar"
  s.source       = { :git => "https://github.com/Kalvar/ios-KRDelta.git", :tag => s.version.to_s }
  s.platform     = :ios, '8.0'
  s.requires_arc = true
  s.public_header_files = 'ML/**/*.h'
  s.source_files = 'ML/**/*.{h,m}'
  s.frameworks   = 'Foundation'
end 