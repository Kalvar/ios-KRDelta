Pod::Spec.new do |s|
  s.name         = "KRDelta"
  s.version      = "1.4.1"
  s.summary      = "KRDelta implemented by Delta Learning Method of Machine Learning that is a supervisor and gradient method."
  s.description  = <<-DESC
                   This classic algorithm could do micro-analysis on mobile as well, we proved its benefit on our mobile businesses.
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