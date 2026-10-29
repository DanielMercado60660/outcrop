//! Minimal Objective-C bridge for render passes and mixed command encoding.
//! outcrop-common exposes compute buffers/pipelines but not render encoders. This
//! local bridge uses only std and the system Objective-C runtime, keeping the
//! crate's sole Rust dependency outcrop-common. All selectors below are Metal APIs.
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr::null_mut;
use outcrop_common::gpu::{Buffer, Library, Pipeline};
use outcrop_common::{Gpu, Result};

type Id = *mut c_void;
#[link(name="objc")]
extern "C" {
    fn objc_getClass(name:*const c_char)->Id;
    fn sel_registerName(name:*const c_char)->Id;
    fn objc_msgSend();
}

// Every call supplies its exact ABI signature, including struct arguments.
// This targets native 64-bit macOS, where Metal's NSUInteger is usize.
macro_rules! send {
    ($object:expr, $selector:literal, $ret:ty $(, $arg:expr => $ty:ty)* $(,)?) => {{
        let method: unsafe extern "C" fn(Id, Id $(, $ty)*) -> $ret =
            std::mem::transmute(objc_msgSend as *const ());
        method($object, sel_registerName(concat!($selector,"\0").as_ptr().cast()) $(, $arg)*)
    }};
}
fn ptr<T>(object:&T)->Id { (object as *const T).cast_mut().cast() }
pub fn check_buffer_lengths(gpu:&Gpu,lengths:&[usize])->Result<()> {
    let maximum=unsafe { send!(ptr(&*gpu.device),"maxBufferLength",usize) };
    for &n in lengths {
        let bytes=n.checked_mul(std::mem::size_of::<f32>())
            .ok_or("Metal buffer size overflow")?;
        if bytes>maximum { return Err(format!("buffer size {bytes} exceeds Metal maximum {maximum}").into()); }
    }
    Ok(())
}
unsafe fn class(name:&str)->Id {
    objc_getClass(CString::new(name).unwrap().as_ptr())
}
unsafe fn string(text:&str)->Id {
    let text=CString::new(text).unwrap();
    send!(class("NSString"),"stringWithUTF8String:",Id,text.as_ptr()=>*const c_char)
}

struct Owned(Id);
impl Owned {
    fn new(object:Id, what:&str)->Result<Self> {
        if object.is_null() { Err(format!("Metal failed to create {what}").into()) }
        else { Ok(Self(object)) }
    }
    unsafe fn retained(object:Id,what:&str)->Result<Self> {
        let object=Self::new(object,what)?;
        send!(object.0,"retain",Id);
        Ok(object)
    }
}
impl Drop for Owned {
    fn drop(&mut self) { unsafe { send!(self.0,"release",()); } }
}
pub struct Pool { _object: Owned }
impl Pool {
    pub fn new()->Self {
        // Releasing this object drains the pool after all invocation resources
        // have been dropped. It is always the handler's first local variable.
        unsafe { Self { _object: Owned(send!(class("NSAutoreleasePool"),"new",Id)) } }
    }
}

pub enum Arg<'a> { Buf(&'a Buffer), Bytes(&'a [u8]) }
impl<'a> Arg<'a> {
    // The only POD types passed here are repr(C) Params and a scalar u32.
    pub fn pod<T: Copy>(value:&'a T)->Self {
        Self::Bytes(unsafe { std::slice::from_raw_parts((value as *const T).cast(),std::mem::size_of::<T>()) })
    }
}
#[repr(C)]
#[derive(Clone,Copy)]
struct Size { width:usize,height:usize,depth:usize }
#[repr(C)]
struct Viewport { x:f64,y:f64,width:f64,height:f64,znear:f64,zfar:f64 }

pub struct Command(Owned);
impl Command {
    pub fn new(gpu:&Gpu)->Result<Self> {
        unsafe { Ok(Self(Owned::retained(send!(ptr(&*gpu.queue),"commandBuffer",Id),"command buffer")?)) }
    }
    unsafe fn encoder(&self)->Result<Owned> {
        Owned::retained(send!(self.0.0,"computeCommandEncoder",Id),"compute encoder")
    }
    unsafe fn dispatch(enc:Id,k:&Pipeline,args:&[Arg],n:usize) {
        send!(enc,"setComputePipelineState:",(),ptr(&**k)=>Id);
        for (i,arg) in args.iter().enumerate() {
            match arg {
                Arg::Buf(b)=>send!(enc,"setBuffer:offset:atIndex:",(),ptr(&***b)=>Id,0=>usize,i=>usize),
                Arg::Bytes(b)=>send!(enc,"setBytes:length:atIndex:",(),b.as_ptr().cast()=>*const c_void,b.len()=>usize,i=>usize),
            }
        }
        let maximum=send!(ptr(&**k),"maxTotalThreadsPerThreadgroup",usize);
        send!(enc,"dispatchThreads:threadsPerThreadgroup:",(),
              Size{width:n,height:1,depth:1}=>Size,
              Size{width:maximum.min(256),height:1,depth:1}=>Size);
    }
    pub fn compute(&self,k:&Pipeline,args:&[Arg],n:usize)->Result<()> {
        unsafe {
            let enc=self.encoder()?;
            Self::dispatch(enc.0,k,args,n);
            send!(enc.0,"endEncoding",());
        }
        Ok(())
    }
    pub fn compute_pair(&self,a:(&Pipeline,&[Arg],usize),b:(&Pipeline,&[Arg],usize))->Result<()> {
        unsafe {
            let enc=self.encoder()?;
            Self::dispatch(enc.0,a.0,a.1,a.2);
            // Default serial compute encoders + tracked resources establish
            // the write/read dependencies between the two dispatches.
            Self::dispatch(enc.0,b.0,b.1,b.2);
            send!(enc.0,"endEncoding",());
        }
        Ok(())
    }
    pub fn compute_texture(&self,k:&Pipeline,args:&[Arg],n:usize,render:&Renderer)->Result<()> {
        unsafe {
            let enc=self.encoder()?;
            send!(enc.0,"setTexture:atIndex:",(),render.texture.0=>Id,0=>usize);
            Self::dispatch(enc.0,k,args,n);
            send!(enc.0,"endEncoding",());
        }
        Ok(())
    }
    pub fn finish(self)->Result<()> {
        unsafe {
            send!(self.0.0,"commit",());
            send!(self.0.0,"waitUntilCompleted",());
            let error=send!(self.0.0,"error",Id);
            if !error.is_null() { return Err(error_text(error).into()); }
            let status=send!(self.0.0,"status",usize);
            if status!=4 { return Err(format!("Metal command status {status}").into()); }
        }
        Ok(())
    }
}
unsafe fn error_text(error:Id)->String {
    if error.is_null() { return "unknown Metal error".into(); }
    let desc=send!(error,"localizedDescription",Id);
    let utf8=send!(desc,"UTF8String",*const c_char);
    if utf8.is_null() { "Metal error without description".into() }
    else { CStr::from_ptr(utf8).to_string_lossy().into_owned() }
}

pub struct Renderer { texture:Owned,pipeline:Owned,pass:Owned,width:usize,height:usize }
impl Renderer {
    pub fn new(gpu:&Gpu,library:&Library,width:usize,height:usize)->Result<Self> {
        unsafe {
            // MTLPixelFormatRGBA32Float=125, 2D=2, Private=2,
            // MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget=1|4.
            let desc=Owned::new(send!(class("MTLTextureDescriptor"),"new",Id),"texture descriptor")?;
            send!(desc.0,"setTextureType:",(),2=>usize);
            send!(desc.0,"setPixelFormat:",(),125=>usize);
            send!(desc.0,"setWidth:",(),width=>usize);
            send!(desc.0,"setHeight:",(),height=>usize);
            send!(desc.0,"setDepth:",(),1=>usize);
            send!(desc.0,"setMipmapLevelCount:",(),1=>usize);
            send!(desc.0,"setArrayLength:",(),1=>usize);
            send!(desc.0,"setSampleCount:",(),1=>usize);
            send!(desc.0,"setStorageMode:",(),2=>usize);
            send!(desc.0,"setUsage:",(),5=>usize);
            let texture=Owned::new(send!(ptr(&*gpu.device),"newTextureWithDescriptor:",Id,desc.0=>Id),"RGBA32Float target")?;
            let vertex=Owned::new(send!(ptr(&**library),"newFunctionWithName:",Id,string("fullscreen")=>Id),"vertex function")?;
            let fragment=Owned::new(send!(ptr(&**library),"newFunctionWithName:",Id,string("color")=>Id),"fragment function")?;
            let descriptor=Owned::new(send!(class("MTLRenderPipelineDescriptor"),"new",Id),"render pipeline descriptor")?;
            send!(descriptor.0,"setVertexFunction:",(),vertex.0=>Id);
            send!(descriptor.0,"setFragmentFunction:",(),fragment.0=>Id);
            send!(descriptor.0,"setSampleCount:",(),1=>usize);
            let attachments=send!(descriptor.0,"colorAttachments",Id);
            let attachment=send!(attachments,"objectAtIndexedSubscript:",Id,0=>usize);
            send!(attachment,"setPixelFormat:",(),125=>usize);
            send!(attachment,"setBlendingEnabled:",(),false=>bool);
            send!(attachment,"setWriteMask:",(),15=>usize);
            let mut error:Id=null_mut();
            let pipeline=send!(ptr(&*gpu.device),"newRenderPipelineStateWithDescriptor:error:",Id,
                               descriptor.0=>Id,&mut error=>*mut Id);
            if pipeline.is_null() { return Err(error_text(error).into()); }
            let pipeline=Owned::new(pipeline,"render pipeline")?;
            let pass=Owned::new(send!(class("MTLRenderPassDescriptor"),"new",Id),"render pass")?;
            let colors=send!(pass.0,"colorAttachments",Id);
            let color=send!(colors,"objectAtIndexedSubscript:",Id,0=>usize);
            send!(color,"setTexture:",(),texture.0=>Id);
            // DontCare load; Store output for subsequent texture read kernels.
            send!(color,"setLoadAction:",(),0=>usize);
            send!(color,"setStoreAction:",(),1=>usize);
            Ok(Self{texture,pipeline,pass,width,height})
        }
    }
    pub fn draw<T:Copy>(&self,command:&Command,state:&Buffer,params:&T)->Result<()> {
        unsafe {
            let enc=Owned::retained(send!(command.0.0,"renderCommandEncoderWithDescriptor:",Id,self.pass.0=>Id),"render encoder")?;
            send!(enc.0,"setRenderPipelineState:",(),self.pipeline.0=>Id);
            send!(enc.0,"setViewport:",(),Viewport{x:0.0,y:0.0,width:self.width as f64,height:self.height as f64,znear:0.0,zfar:1.0}=>Viewport);
            send!(enc.0,"setFragmentBuffer:offset:atIndex:",(),ptr(&**state)=>Id,0=>usize,0=>usize);
            send!(enc.0,"setFragmentBytes:length:atIndex:",(),ptr(params) as *const c_void=>*const c_void,
                  std::mem::size_of::<T>()=>usize,1=>usize);
            // MTLPrimitiveTypeTriangle=3; one oversized triangle covers target.
            send!(enc.0,"drawPrimitives:vertexStart:vertexCount:",(),3=>usize,0=>usize,3=>usize);
            send!(enc.0,"endEncoding",());
        }
        Ok(())
    }
}
