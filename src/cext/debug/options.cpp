#include <c10/util/Exception.h>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string_view>

#include <xpool/debug/options.hpp>

namespace xpool::debug {

NLOHMANN_JSON_SERIALIZE_ENUM(LoopbackSite::Type, {{LoopbackSite::None, nullptr},
                                                  {LoopbackSite::Instance, "instance"},
                                                  {LoopbackSite::AtnAgent, "atnagent"},
                                                  {LoopbackSite::FfnAgent, "ffnagent"}})

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LoopbackOptions, enable, site)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TraceOptions, enable, trace_capacity)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DebugOptions, loopback, transport_observer, fabric_observer)

DebugOptions DebugOptions::parse(std::string_view json) {
  try {
    const nlohmann::json value = nlohmann::json::parse(json);
    TORCH_CHECK(value.is_object() && value.size() == 3U, "xpool debug options must be an object with exactly three "
                                                         "sections");

    const auto &loopback = value.at("loopback");
    TORCH_CHECK(loopback.is_object() && loopback.size() == 2U,
                "xpool debug loopback section must be an object with exactly "
                "two fields");

    for (const char *name : {"transport_observer", "fabric_observer"}) {
      const auto &observer = value.at(name);
      TORCH_CHECK(observer.is_object() && observer.size() == 2U, "xpool debug ", name,
                  " section must be an object with exactly two fields");
      const auto &capacity = observer.at("trace_capacity");
      TORCH_CHECK(capacity.is_number_integer(), "xpool debug trace_capacity must be an integer");
      TORCH_CHECK(capacity.is_number_unsigned() ? capacity.get<std::size_t>() > 0U : capacity.get<std::int64_t>() > 0,
                  "xpool debug trace_capacity must be positive");
    }

    auto options = value.get<DebugOptions>();
    const auto &site = loopback.at("site");
    TORCH_CHECK(options.loopback.site != LoopbackSite::None || site.is_null(),
                "xpool debug loopback.site is invalid: ", site.dump());
    TORCH_CHECK(options.loopback.enabled() == (options.loopback.site != LoopbackSite::None),
                "xpool loopback enable and site must agree");
    return options;
  } catch (const nlohmann::json::exception &error) {
    TORCH_CHECK(false, "xpool native debug JSON is invalid: ", error.what());
  }
}

} // namespace xpool::debug
